import base64
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routers import dietician, gym_recommender, gym_trainer, habit_tracker, pose_analyzer, smart_gym, virtual_buddy


st.set_page_config(
    page_title="AI Gym & Fitness Assistant",
    page_icon="⚡",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 10px;
        padding: 1rem;
        background: rgba(250, 250, 250, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_json_block(title, data):
    st.subheader(title)
    st.json(data)


def render_trainer():
    st.header("AI Gym Trainer")
    st.caption("MediaPipe-based posture analysis, rep phase tracking, form score, and annotated visual feedback.")

    health = gym_trainer.get_pose_runtime_status()
    if health["status"] == "ok":
        st.success("Pose runtime ready: MediaPipe + OpenCV + model loaded.")
    else:
        st.error(health["message"])
        st.info("On Streamlit Cloud, confirm `backend/models/pose_landmarker_lite.task` is committed and `requirements.txt` installs successfully.")

    left, right = st.columns([1, 1])
    with left:
        exercise = st.selectbox("Exercise", ["squat", "pushup", "bicep_curl"], format_func=lambda v: v.replace("_", " ").title())
        source = st.radio("Input", ["Camera", "Upload image"], horizontal=True)

        if "trainer_reps" not in st.session_state:
            st.session_state.trainer_reps = 0
        if "trainer_phase" not in st.session_state:
            st.session_state.trainer_phase = None

        if st.button("Reset reps"):
            st.session_state.trainer_reps = 0
            st.session_state.trainer_phase = None
            st.session_state.pop("trainer_result", None)

        image_file = None
        if source == "Camera":
            image_file = st.camera_input("Capture a clear full-body frame")
        else:
            image_file = st.file_uploader("Upload a clear full-body frame", type=["jpg", "jpeg", "png"])

        if image_file and st.button("Analyze frame", type="primary"):
            try:
                result = gym_trainer.analyze_image_bytes(
                    image_file.getvalue(),
                    exercise,
                    st.session_state.trainer_phase,
                    st.session_state.trainer_reps,
                )
                data = result.model_dump()
                st.session_state.trainer_result = data
                st.session_state.trainer_reps = data["reps_counted"]
                st.session_state.trainer_phase = data["phase"]
            except Exception as exc:
                st.error(str(exc))

    with right:
        result = st.session_state.get("trainer_result")
        if result:
            c1, c2, c3 = st.columns(3)
            c1.metric("Form score", result["form_score"])
            c2.metric("Reps", result["reps_counted"])
            c3.metric("Confidence", f"{round(result['confidence'] * 100)}%")
            st.write("Phase:", result["phase"] or "ready")
            for item in result["feedback"]:
                st.write("- " + item)
            if result["joint_angles"]:
                st.write("Joint angles")
                st.json(result["joint_angles"])
            if result["annotated_image_b64"]:
                image_bytes = base64.b64decode(result["annotated_image_b64"])
                st.image(image_bytes, caption="Annotated pose feedback", use_container_width=True)
        else:
            st.info("Capture or upload a frame to see trainer feedback.")

    with st.expander("Workout plan"):
        target_reps = st.number_input("Target reps", min_value=1, max_value=50, value=10)
        target_sets = st.number_input("Target sets", min_value=1, max_value=10, value=3)
        if st.button("Generate workout plan"):
            plan = gym_trainer.generate_workout_plan(
                gym_trainer.WorkoutSession(exercise=exercise, target_reps=int(target_reps), target_sets=int(target_sets))
            )
            st.json(plan)


def render_dietician():
    st.header("AI Dietician & Calorie Coach")
    with st.form("diet_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", "Lavanya")
        age = c2.number_input("Age", min_value=12, max_value=90, value=20)
        gender = c3.selectbox("Gender", ["female", "male"])
        c4, c5 = st.columns(2)
        height = c4.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=162.0)
        weight = c5.number_input("Weight (kg)", min_value=25.0, max_value=200.0, value=55.0)
        goal = st.selectbox("Goal", ["maintain", "lose_weight", "gain_muscle"])
        activity = st.selectbox("Activity level", ["sedentary", "light", "moderate", "active", "very_active"], index=2)
        preference = st.selectbox("Diet preference", ["vegetarian", "vegan", "non-vegetarian", "keto"])
        submitted = st.form_submit_button("Create diet plan", type="primary")

    profile = dietician.UserProfile(
        name=name,
        age=int(age),
        gender=gender,
        height_cm=float(height),
        weight_kg=float(weight),
        goal=goal,
        activity_level=activity,
        dietary_preference=preference,
        allergies=[],
    )
    if submitted:
        show_json_block("Diet plan", dietician.generate_diet_plan(profile))

    if st.button("Generate grocery list"):
        show_json_block("Weekly grocery list", dietician.generate_grocery_list(profile))


def render_smart_gym():
    st.header("Smart Gym Assistant")
    c1, c2, c3 = st.columns(3)
    heart_rate = c1.slider("Heart rate", 60, 210, 132)
    resistance = c2.slider("Resistance", 0, 100, 45)
    speed = c3.slider("Speed (km/h)", 0.0, 20.0, 7.5)
    session_minutes = st.slider("Session minutes", 1, 120, 35)
    level = st.selectbox("Fitness level", ["beginner", "intermediate", "advanced"], index=1)
    data = smart_gym.IoTSensorData(
        equipment_id="TM-01",
        equipment_type="treadmill",
        current_resistance=float(resistance),
        current_speed=float(speed),
        heart_rate=int(heart_rate),
        calories_burned=float(session_minutes * 8),
        session_minutes=float(session_minutes),
        user_fitness_level=level,
    )
    if st.button("Monitor performance", type="primary"):
        show_json_block("Monitoring result", smart_gym.monitor_performance(data))
    if st.button("Show equipment status"):
        show_json_block("Equipment status", smart_gym.get_equipment_status())


def render_habit_tracker():
    st.header("AI Fitness Habit Tracker")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("User name", "Lavanya")
    streak = c2.number_input("Current streak", min_value=0, value=5)
    missed = c3.number_input("Missed last week", min_value=0, max_value=7, value=1)
    stress = st.slider("Stress level", 1, 10, 4)
    sleep = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
    style = st.selectbox("Motivation style", ["data", "inspirational", "competitive"])
    days = st.multiselect("Workout days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ["Monday", "Wednesday", "Friday"])
    profile = habit_tracker.HabitProfile(
        user_id="user1",
        name=name,
        workout_days=days,
        preferred_time="morning",
        streak_days=int(streak),
        total_workouts=22,
        missed_last_week=int(missed),
        stress_level=int(stress),
        sleep_hours=float(sleep),
        motivation_style=style,
    )
    if st.button("Predict skip risk", type="primary"):
        show_json_block("Skip risk", habit_tracker.predict_skip(profile))
    if st.button("Leaderboard"):
        show_json_block("Weekly leaderboard", habit_tracker.get_leaderboard("weekly"))


def render_virtual_buddy():
    st.header("Virtual Gym Buddy")
    context = st.selectbox("Context", ["general", "pre_workout", "post_workout"])
    message = st.text_area("Message", "How much protein should I eat?")
    if st.button("Send", type="primary"):
        response = virtual_buddy.chat(virtual_buddy.ChatMessage(user_id="user1", message=message, context=context))
        st.success(response["response"])
        st.caption(f"Detected sentiment: {response['sentiment']}")
    if st.button("Daily quote"):
        show_json_block("Quote", virtual_buddy.get_daily_quote())


def render_pose_analyzer():
    st.header("Pose-to-Performance Analyzer")
    exercise = st.selectbox("Exercise", ["squat", "pushup", "bicep_curl", "deadlift"])
    sets = st.number_input("Sets completed", min_value=1, max_value=10, value=3)
    reps_text = st.text_input("Reps per set", "10,9,10")
    scores_text = st.text_input("Form scores", "82,78,85")
    duration = st.number_input("Duration seconds", min_value=10, value=420)
    if st.button("Analyze session", type="primary"):
        reps = [int(x.strip()) for x in reps_text.split(",") if x.strip()]
        scores = [float(x.strip()) for x in scores_text.split(",") if x.strip()]
        session = pose_analyzer.PoseSession(
            user_id="user1",
            exercise=exercise,
            sets_completed=int(sets),
            reps_per_set=reps,
            form_scores=scores,
            duration_seconds=int(duration),
        )
        show_json_block("Performance analysis", pose_analyzer.analyze_session(session))
    if st.button("Efficiency tips"):
        show_json_block("Tips", pose_analyzer.get_efficiency_tips(exercise))


def render_gym_recommender():
    st.header("Gym Recommender & Planner")
    city = st.selectbox("City", ["pune", "mumbai"])
    goal = st.selectbox("Goal", ["weight_loss", "muscle_gain", "endurance", "flexibility"])
    level = st.selectbox("Fitness level", ["beginner", "intermediate", "advanced"])
    budget = st.number_input("Monthly budget INR", min_value=500, max_value=10000, value=2000)
    pref = gym_recommender.UserPreference(
        user_id="user1",
        location=city,
        fitness_goal=goal,
        fitness_level=level,
        budget_inr=int(budget),
        preferred_timing="morning",
        equipment_preference="gym",
        past_programs=[],
    )
    if st.button("Recommend gyms", type="primary"):
        show_json_block("Gym recommendations", gym_recommender.recommend_gyms(pref))
    if st.button("Recommend workout program"):
        req = gym_recommender.ProgramRequest(fitness_goal=goal, fitness_level=level, days_per_week=3, equipment="full_gym")
        show_json_block("Program", gym_recommender.recommend_program(req))
    if st.button("Show challenges"):
        show_json_block("Challenges", gym_recommender.get_challenges())


st.title("AI Gym & Fitness Assistant")
st.caption("Streamlit deployment version for the full AI fitness project.")

tabs = st.tabs([
    "Gym Trainer",
    "Dietician",
    "Smart Gym",
    "Habit Tracker",
    "Virtual Buddy",
    "Pose Analyzer",
    "Gym Recommender",
])

with tabs[0]:
    render_trainer()
with tabs[1]:
    render_dietician()
with tabs[2]:
    render_smart_gym()
with tabs[3]:
    render_habit_tracker()
with tabs[4]:
    render_virtual_buddy()
with tabs[5]:
    render_pose_analyzer()
with tabs[6]:
    render_gym_recommender()
