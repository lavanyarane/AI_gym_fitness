"""
Module 1: AI Gym Trainer — Workout Detection & Feedback System
Uses MediaPipe/OpenCV for pose detection, rep counting, and form correction.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import base64
from pathlib import Path

router = APIRouter()
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
SUPPORTED_EXERCISES = {"squat", "pushup", "bicep_curl"}
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"
_TASK_LANDMARKER = None
MEDIAPIPE_IMPORT_ERROR = None
CV2_IMPORT_ERROR = None

# ── Safe MediaPipe import (new API for 0.10+) ──
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    try:
        _pose_solution = mp.solutions.pose
        _drawing       = mp.solutions.drawing_utils
        MEDIAPIPE_MODE  = "legacy"
    except AttributeError:
        MEDIAPIPE_MODE = "tasks"

    MEDIAPIPE_AVAILABLE = True
except ImportError as exc:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_MODE      = "none"
    MEDIAPIPE_IMPORT_ERROR = str(exc)

# ── OpenCV import ──
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as exc:
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(exc)


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
class WorkoutSession(BaseModel):
    exercise: str
    target_reps: int = 10
    target_sets: int = 3

class FeedbackResponse(BaseModel):
    exercise: str
    reps_counted: int
    form_score: float
    feedback: List[str]
    joint_angles: dict
    annotated_image_b64: Optional[str] = None
    phase: Optional[str] = None
    confidence: float = 0.0
    person_detected: bool = False
    mode: str = "mediapipe_live"


# ──────────────────────────────────────────────
# Angle calculator
# ──────────────────────────────────────────────
def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle


# ──────────────────────────────────────────────
# Rule-based form analyzers (pure math, no MP needed)
# ──────────────────────────────────────────────
def analyze_squat_angles(knee_angle, hip_angle):
    feedback, score = [], 100.0
    if knee_angle > 105:
        feedback.append("Go deeper until your thighs are close to parallel.")
        score -= 25
    elif knee_angle < 55:
        feedback.append("You are dropping too deep; stop near parallel to protect your knees.")
        score -= 12
    if hip_angle < 65:
        feedback.append("Lift your chest and keep your spine more neutral.")
        score -= 18
    if not feedback:
        feedback.append("Strong squat posture. Keep the same depth and torso control.")
    return {"knee_angle": round(knee_angle,1), "hip_angle": round(hip_angle,1),
            "feedback": feedback, "score": max(score, 0)}

def analyze_pushup_angles(elbow_angle, body_angle):
    feedback, score = [], 100.0
    if elbow_angle > 160:
        feedback.append("Lower your chest closer to the ground for a full rep.")
        score -= 20
    if body_angle < 160:
        feedback.append("Keep your shoulders, hips, and knees in one straight line.")
        score -= 15
    if not feedback:
        feedback.append("Clean push-up line and range of motion.")
    return {"elbow_angle": round(elbow_angle,1), "body_angle": round(body_angle,1),
            "feedback": feedback, "score": max(score, 0)}

def analyze_curl_angles(elbow_angle):
    feedback, score = [], 100.0
    if elbow_angle > 155:
        feedback.append("Curl the weight upward through a fuller range of motion.")
        score -= 20
    if elbow_angle < 35:
        feedback.append("Control the top position; avoid shortening the rep by swinging.")
        score -= 10
    if not feedback:
        feedback.append("Good curl control. Keep elbows pinned and lower slowly.")
    return {"elbow_angle": round(elbow_angle,1), "feedback": feedback, "score": max(score, 0)}


def update_rep_state(exercise: str, angles: dict, previous_phase: Optional[str], reps_counted: int) -> tuple[str, int]:
    """Count a rep only when the movement crosses a full down/up threshold."""
    reps = max(reps_counted, 0)
    phase = previous_phase or "ready"

    if exercise == "squat":
        knee = angles["knee_angle"]
        if knee < 100:
            phase = "down"
        elif previous_phase == "down" and knee > 150:
            reps += 1
            phase = "up"
        elif knee > 150:
            phase = "up"
    elif exercise == "pushup":
        elbow = angles["elbow_angle"]
        if elbow < 100:
            phase = "down"
        elif previous_phase == "down" and elbow > 155:
            reps += 1
            phase = "up"
        elif elbow > 155:
            phase = "up"
    else:
        elbow = angles["elbow_angle"]
        if elbow > 145:
            phase = "down"
        elif previous_phase == "down" and elbow < 70:
            reps += 1
            phase = "up"
        elif elbow < 70:
            phase = "up"

    return phase, reps


def apply_session_context(
    exercise: str,
    analysis: dict,
    joint_angles: dict,
    phase: str,
    previous_phase: Optional[str],
    reps_counted: int
) -> dict:
    """Avoid awarding perfect scores for static poses that merely resemble an exercise phase."""
    feedback = list(analysis["feedback"])
    score = float(analysis["score"])
    has_movement_context = bool(previous_phase and previous_phase != "ready") or reps_counted > 0

    if exercise == "squat":
        knee = joint_angles["knee_angle"]
        hip = joint_angles["hip_angle"]
        if not has_movement_context and knee <= 110:
            score = min(score, 55)
            feedback.insert(0, "Start from a tall standing position, then squat down. A seated or already-bottom position is not counted as a valid squat.")
        elif phase == "up" and reps_counted == 0:
            score = min(score, 70)
            feedback.insert(0, "Standing position detected. Begin the squat by lowering under control.")
        if 70 <= knee <= 110 and 75 <= hip <= 120 and not has_movement_context:
            score = min(score, 50)
            feedback.insert(0, "This looks like a static seated/deep-knee posture. Stand up fully first so the trainer can verify the full movement.")

    elif exercise == "pushup":
        elbow = joint_angles["elbow_angle"]
        if not has_movement_context and elbow <= 110:
            score = min(score, 60)
            feedback.insert(0, "Start from a straight-arm plank, then lower and press back up. A single bottom-frame is not a complete push-up.")
        elif phase == "up" and reps_counted == 0:
            score = min(score, 75)
            feedback.insert(0, "Top plank detected. Lower your chest to begin the rep.")

    else:
        elbow = joint_angles["elbow_angle"]
        if not has_movement_context and elbow < 80:
            score = min(score, 60)
            feedback.insert(0, "Start with the arm extended, then curl up. A held top position is not a complete curl.")
        elif phase == "down" and reps_counted == 0:
            score = min(score, 75)
            feedback.insert(0, "Starting position detected. Curl upward under control to begin the rep.")

    deduped_feedback = []
    for item in feedback:
        if item not in deduped_feedback:
            deduped_feedback.append(item)

    return {**analysis, "feedback": deduped_feedback, "score": round(max(score, 0), 1)}


def build_pose_response(frame_bgr, exercise: str, landmarks, previous_phase: Optional[str], reps_counted: int, draw_connections=None):
    def pt(idx):
        landmark = landmarks[idx]
        return [landmark.x, landmark.y]

    def visibility(indices):
        values = [getattr(landmarks[i], "visibility", 1.0) or 0.0 for i in indices]
        return min(values)

    PL = mp_vision.PoseLandmark
    left_leg = [PL.LEFT_HIP.value, PL.LEFT_KNEE.value, PL.LEFT_ANKLE.value, PL.LEFT_SHOULDER.value]
    right_leg = [PL.RIGHT_HIP.value, PL.RIGHT_KNEE.value, PL.RIGHT_ANKLE.value, PL.RIGHT_SHOULDER.value]
    use_left = visibility(left_leg) >= visibility(right_leg)

    if exercise == "squat":
        hip = PL.LEFT_HIP.value if use_left else PL.RIGHT_HIP.value
        knee = PL.LEFT_KNEE.value if use_left else PL.RIGHT_KNEE.value
        ankle = PL.LEFT_ANKLE.value if use_left else PL.RIGHT_ANKLE.value
        shoulder = PL.LEFT_SHOULDER.value if use_left else PL.RIGHT_SHOULDER.value
        knee_a = calculate_angle(pt(hip), pt(knee), pt(ankle))
        hip_a = calculate_angle(pt(shoulder), pt(hip), pt(knee))
        analysis = analyze_squat_angles(knee_a, hip_a)
    elif exercise == "pushup":
        left_arm = [PL.LEFT_SHOULDER.value, PL.LEFT_ELBOW.value, PL.LEFT_WRIST.value, PL.LEFT_HIP.value, PL.LEFT_KNEE.value]
        right_arm = [PL.RIGHT_SHOULDER.value, PL.RIGHT_ELBOW.value, PL.RIGHT_WRIST.value, PL.RIGHT_HIP.value, PL.RIGHT_KNEE.value]
        use_left = visibility(left_arm) >= visibility(right_arm)
        shoulder = PL.LEFT_SHOULDER.value if use_left else PL.RIGHT_SHOULDER.value
        elbow = PL.LEFT_ELBOW.value if use_left else PL.RIGHT_ELBOW.value
        wrist = PL.LEFT_WRIST.value if use_left else PL.RIGHT_WRIST.value
        hip = PL.LEFT_HIP.value if use_left else PL.RIGHT_HIP.value
        knee = PL.LEFT_KNEE.value if use_left else PL.RIGHT_KNEE.value
        el_a = calculate_angle(pt(shoulder), pt(elbow), pt(wrist))
        body_a = calculate_angle(pt(shoulder), pt(hip), pt(knee))
        analysis = analyze_pushup_angles(el_a, body_a)
    else:
        left_arm = [PL.LEFT_SHOULDER.value, PL.LEFT_ELBOW.value, PL.LEFT_WRIST.value]
        right_arm = [PL.RIGHT_SHOULDER.value, PL.RIGHT_ELBOW.value, PL.RIGHT_WRIST.value]
        use_left = visibility(left_arm) >= visibility(right_arm)
        shoulder = PL.LEFT_SHOULDER.value if use_left else PL.RIGHT_SHOULDER.value
        elbow = PL.LEFT_ELBOW.value if use_left else PL.RIGHT_ELBOW.value
        wrist = PL.LEFT_WRIST.value if use_left else PL.RIGHT_WRIST.value
        el_a = calculate_angle(pt(shoulder), pt(elbow), pt(wrist))
        analysis = analyze_curl_angles(el_a)

    joint_angles = {k: v for k, v in analysis.items() if "angle" in k}
    phase, reps = update_rep_state(exercise, joint_angles, previous_phase, reps_counted)
    analysis = apply_session_context(exercise, analysis, joint_angles, phase, previous_phase, reps_counted)
    confidence = round(float(np.mean([getattr(landmark, "visibility", 1.0) or 0.0 for landmark in landmarks])), 3)

    if draw_connections:
        draw_connections(frame_bgr, landmarks)
    cv2.putText(frame_bgr, f"{exercise} reps: {reps} phase: {phase}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"score: {analysis['score']:.0f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", frame_bgr)
    b64 = base64.b64encode(buf).decode("utf-8")
    return analysis, joint_angles, phase, reps, confidence, b64


def get_task_landmarker():
    global _TASK_LANDMARKER
    if _TASK_LANDMARKER is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"Pose model is missing at {MODEL_PATH}")
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _TASK_LANDMARKER = mp_vision.PoseLandmarker.create_from_options(options)
    return _TASK_LANDMARKER


def draw_task_landmarks(frame_bgr, landmarks):
    height, width = frame_bgr.shape[:2]
    for connection in mp_vision.PoseLandmarksConnections.POSE_LANDMARKS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        if (getattr(start, "visibility", 1.0) or 0.0) < 0.45 or (getattr(end, "visibility", 1.0) or 0.0) < 0.45:
            continue
        start_xy = (int(start.x * width), int(start.y * height))
        end_xy = (int(end.x * width), int(end.y * height))
        cv2.line(frame_bgr, start_xy, end_xy, (0, 220, 255), 2)
    for landmark in landmarks:
        if (getattr(landmark, "visibility", 1.0) or 0.0) < 0.45:
            continue
        cv2.circle(frame_bgr, (int(landmark.x * width), int(landmark.y * height)), 4, (0, 255, 120), -1)


# ──────────────────────────────────────────────
# MediaPipe legacy (0.9.x) analyzer
# ──────────────────────────────────────────────
def analyze_with_legacy_mp(frame_bgr, exercise, previous_phase: Optional[str], reps_counted: int):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with _pose_solution.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark

        def draw_legacy_landmarks(image, _landmarks):
            _drawing.draw_landmarks(image, results.pose_landmarks, _pose_solution.POSE_CONNECTIONS)

        return build_pose_response(frame_bgr, exercise, lm, previous_phase, reps_counted, draw_legacy_landmarks)


def analyze_with_tasks_mp(frame_bgr, exercise, previous_phase: Optional[str], reps_counted: int):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = get_task_landmarker().detect(mp_image)
    if not result.pose_landmarks:
        return None
    return build_pose_response(frame_bgr, exercise, result.pose_landmarks[0], previous_phase, reps_counted, draw_task_landmarks)


def analyze_image_bytes(contents: bytes, exercise: str, previous_phase: Optional[str], reps_counted: int):
    if not MEDIAPIPE_AVAILABLE or not CV2_AVAILABLE or MEDIAPIPE_MODE == "none":
        raise HTTPException(503, get_pose_runtime_status()["message"])

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Uploaded image could not be decoded")

    if MEDIAPIPE_MODE == "legacy":
        result = analyze_with_legacy_mp(frame, exercise, previous_phase, reps_counted)
    else:
        result = analyze_with_tasks_mp(frame, exercise, previous_phase, reps_counted)
    if result is None:
        return FeedbackResponse(
            exercise=exercise,
            reps_counted=max(reps_counted, 0),
            form_score=0,
            feedback=["No full body pose detected. Step back, improve lighting, and keep the whole movement visible."],
            joint_angles={},
            phase=previous_phase,
            confidence=0,
            person_detected=False,
            mode="no_pose"
        )

    analysis, joint_angles, phase, reps, confidence, b64 = result
    return FeedbackResponse(
        exercise=exercise,
        reps_counted=reps,
        form_score=analysis["score"],
        feedback=analysis["feedback"],
        joint_angles=joint_angles,
        annotated_image_b64=b64,
        phase=phase,
        confidence=confidence,
        person_detected=True,
        mode="mediapipe_live"
    )


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@router.post("/analyze-frame", response_model=FeedbackResponse)
async def analyze_frame(
    exercise: str,
    previous_phase: Optional[str] = Query(default=None),
    reps_counted: int = Query(default=0, ge=0),
    file: UploadFile = File(...)
):
    """
    Analyze a single video frame for pose detection and form feedback.
    Works with or without MediaPipe installed.
    """
    exercise = exercise.lower().strip()
    if exercise not in SUPPORTED_EXERCISES:
        raise HTTPException(400, f"Exercise '{exercise}' not supported. Choose: {sorted(SUPPORTED_EXERCISES)}")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image uploads are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Uploaded image is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Uploaded image must be 5 MB or smaller")

    return analyze_image_bytes(contents, exercise, previous_phase, reps_counted)


@router.post("/generate-plan")
def generate_workout_plan(session: WorkoutSession):
    """Generate a personalized workout plan."""
    CUES = {
        "squat":      "Feet shoulder-width, chest up, push knees out",
        "pushup":     "Straight body, elbows 45°, full range",
        "bicep_curl": "Elbows pinned, slow eccentric, full extension",
    }
    ex = session.exercise.lower()
    if ex not in CUES:
        raise HTTPException(400, "Exercise not found")

    plan = [
        {"set": i+1, "reps": session.target_reps,
         "rest_seconds": 60, "cue": CUES[ex]}
        for i in range(session.target_sets)
    ]
    return {
        "exercise": ex,
        "total_sets": session.target_sets,
        "total_reps": session.target_reps * session.target_sets,
        "estimated_time_minutes": session.target_sets * 2,
        "plan": plan,
        "warm_up":   ["5 min light cardio", "10 dynamic stretches", "1 set at 50% weight"],
        "cool_down": ["Static stretching 5 min", "Foam rolling if available"]
    }


@router.get("/exercises")
def list_exercises():
    return {
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "mediapipe_mode": MEDIAPIPE_MODE,
        "supported_exercises": [
            {"id": "squat",      "name": "Squat",      "muscles": ["Quads","Glutes","Hamstrings"]},
            {"id": "pushup",     "name": "Push-Up",    "muscles": ["Chest","Triceps","Shoulders"]},
            {"id": "bicep_curl", "name": "Bicep Curl", "muscles": ["Biceps","Forearms"]},
        ]
    }


@router.get("/health")
def health():
    return get_pose_runtime_status()


def get_pose_runtime_status():
    missing = []
    if not MEDIAPIPE_AVAILABLE:
        missing.append(f"mediapipe import failed: {MEDIAPIPE_IMPORT_ERROR or 'not installed'}")
    if not CV2_AVAILABLE:
        missing.append(f"opencv import failed: {CV2_IMPORT_ERROR or 'not installed'}")
    if not MODEL_PATH.exists():
        missing.append(f"pose model missing: {MODEL_PATH}")

    ready = MEDIAPIPE_AVAILABLE and CV2_AVAILABLE and MEDIAPIPE_MODE != "none" and MODEL_PATH.exists()
    return {
        "status": "ok" if ready else "unavailable",
        "mediapipe": MEDIAPIPE_MODE,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "mediapipe_error": MEDIAPIPE_IMPORT_ERROR,
        "opencv": CV2_AVAILABLE,
        "opencv_error": CV2_IMPORT_ERROR,
        "model_present": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "missing": missing,
        "message": "Full pose detection active" if ready else "Live pose detection unavailable: " + "; ".join(missing)
    }
