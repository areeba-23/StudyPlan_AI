import traceback
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

import joblib
import numpy as np
import pandas as pd
import shap
import dice_ml
from dice_ml import Dice
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.main { background: #0f0f17; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    border: 1px solid #2a2a4a;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e2e8f0;
}

.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 28px 0 16px 0;
    padding: 12px 16px;
    background: linear-gradient(90deg, #1e293b, transparent);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
}

.factor-row {
    display: flex;
    align-items: center;
    margin: 10px 0;
    gap: 12px;
    padding: 8px 12px;
    background: #1a1a2e;
    border-radius: 8px;
}

.factor-label {
    width: 210px;
    font-size: 0.82rem;
    color: #cbd5e1;
    line-height: 1.3;
}

.factor-bar-bg {
    flex: 1;
    background: #2d2d4e;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}

.factor-pts {
    width: 65px;
    font-size: 0.82rem;
    font-weight: 600;
    text-align: right;
}

.plan-card {
    background: linear-gradient(160deg, #1a1a2e 0%, #151528 100%);
    border-radius: 16px;
    padding: 22px 18px;
    border: 1px solid #2a2a4a;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    min-height: 280px;
}

.plan-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #2a2a4a;
}

.plan-item {
    background: #0f1729;
    border-radius: 8px;
    padding: 10px 12px;
    margin: 8px 0;
    font-size: 0.82rem;
    color: #94a3b8;
    border: 1px solid #1e293b;
}

.plan-item b {
    color: #e2e8f0;
    font-size: 0.9rem;
}

.plan-item .arrow-up { color: #4ade80; }
.plan-item .arrow-down { color: #f87171; }

.plan-score {
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 14px;
    font-weight: 700;
    text-align: center;
    font-size: 1rem;
    letter-spacing: 0.02em;
}

.warning-box {
    background: #2d1f0e;
    border: 1px solid #92400e;
    border-radius: 10px;
    padding: 12px 16px;
    color: #fbbf24;
    font-size: 0.88rem;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────
try:
    model = joblib.load('student_performance_model.pkl')
    feature_names = joblib.load('feature_names.pkl')

except Exception as e:
    st.error(f"❌ Model load error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ── Session state init ───────────────────────────────────
for key in [
    'predicted_score', 'grade', 'emoji', 'input_df', 'shap_values',
    'show_plans', 'selected_range', 's_hours', 's_attendance',
    's_sleep', 's_tutoring', 's_physical', 's_motivation',
    's_parental', 's_access', 's_peer', 's_teacher', 's_family',
    's_disability', 's_internet', 's_extracurricular'
]:
    if key not in st.session_state:
        st.session_state[key] = None

if 'show_plans' not in st.session_state:
    st.session_state.show_plans = False

st.title(" Student Performance Predictor")

st.markdown(
    "Fill in your details below to get your predicted score and a personalized improvement plan."
)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("📋 Student Information")

hours_studied = st.sidebar.slider("Hours Studied per Week", 1, 44, 20)
attendance = st.sidebar.slider("Attendance (%)", 60, 100, 80)
sleep_hours = st.sidebar.slider("Sleep Hours per Night", 4, 10, 7)
previous_scores = st.sidebar.slider("Previous Scores", 50, 100, 75)
tutoring_sessions = st.sidebar.slider("Tutoring Sessions per Month", 0, 8, 2)
physical_activity = st.sidebar.slider("Physical Activity (hrs/week)", 0, 6, 3)

st.sidebar.markdown("---")

parental_involvement = st.sidebar.selectbox(
    "Parental Involvement",
    ["Low", "Medium", "High"]
)

access_to_resources = st.sidebar.selectbox(
    "Access to Resources",
    ["Low", "Medium", "High"]
)

motivation_level = st.sidebar.selectbox(
    "Motivation Level",
    ["Low", "Medium", "High"]
)

family_income = st.sidebar.selectbox(
    "Family Income",
    ["Low", "Medium", "High"]
)

teacher_quality = st.sidebar.selectbox(
    "Teacher Quality",
    ["Low", "Medium", "High"]
)

peer_influence = st.sidebar.selectbox(
    "Peer Influence",
    ["Negative", "Neutral", "Positive"]
)

distance_from_home = st.sidebar.selectbox(
    "Distance from Home",
    ["Near", "Moderate", "Far"]
)

parental_education = st.sidebar.selectbox(
    "Parental Education Level",
    ["High School", "College", "Postgraduate"]
)

st.sidebar.markdown("---")

extracurricular = st.sidebar.checkbox(
    "Extracurricular Activities",
    value=True
)

internet_access = st.sidebar.checkbox(
    "Internet Access",
    value=True
)

learning_disability = st.sidebar.checkbox(
    "Learning Disabilities",
    value=False
)

gender_male = st.sidebar.checkbox(
    "Gender: Male",
    value=True
)

# ── Feature engineering ──────────────────────────────────
def build_features(
    hs, att, sl, ps, ts, pa,
    pi, ar, ml, fi, tq, peer,
    dfh, pe, ec, ia, ld, gm
):

    return pd.DataFrame([{
        'Hours_Studied': hs,
        'Attendance': att,
        'Sleep_Hours': sl,
        'Previous_Scores': ps,
        'Tutoring_Sessions': ts,
        'Physical_Activity': pa,

        'Parental_Involvement_Low': int(pi == "Low"),
        'Parental_Involvement_Medium': int(pi == "Medium"),

        'Access_to_Resources_Low': int(ar == "Low"),
        'Access_to_Resources_Medium': int(ar == "Medium"),

        'Extracurricular_Activities_Yes': int(ec),

        'Motivation_Level_Low': int(ml == "Low"),
        'Motivation_Level_Medium': int(ml == "Medium"),

        'Internet_Access_Yes': int(ia),

        'Family_Income_Low': int(fi == "Low"),
        'Family_Income_Medium': int(fi == "Medium"),

        'Teacher_Quality_Low': int(tq == "Low"),
        'Teacher_Quality_Medium': int(tq == "Medium"),

        'School_Type_Public': 1,

        'Peer_Influence_Neutral': int(peer == "Neutral"),
        'Peer_Influence_Positive': int(peer == "Positive"),

        'Learning_Disabilities_Yes': int(ld),

        'Parental_Education_Level_High School': int(pe == "High School"),
        'Parental_Education_Level_Postgraduate': int(pe == "Postgraduate"),

        'Distance_from_Home_Moderate': int(dfh == "Moderate"),
        'Distance_from_Home_Near': int(dfh == "Near"),

        'Gender_Male': int(gm),

        'Study_Efficiency': hs * ps,
        'Wellness': sl * pa,
        'Study_Intensity': hs / (sl + 1),
        'Attendance_Study_Balance': att * hs,
        'Score_Consistency': ps * att / 100,

        'Risk_Factor': int(ld) + int(pi == "Low") + int(ml == "Low"),

        'Motivation_Adjusted_Study': hs * (
            3 if ml == "High"
            else 2 if ml == "Medium"
            else 1
        ),
    }])

def score_to_grade(score):

    if score >= 80:
        return "A", "🟢"

    elif score >= 70:
        return "B", "🔵"

    elif score >= 60:
        return "C", "🟡"

    elif score >= 50:
        return "D", "🟠"

    else:
        return "F", "🔴"

def get_grade_options(grade):

    if grade == "D":
        return {
            "Grade C (60-69) ✅ Recommended": [60, 69]
        }

    elif grade == "C":
        return {
            "Grade B (70-79) ✅ Recommended": [70, 79],
            "Grade A (80+) ⚠️ Challenging": [80, 92],
        }

    elif grade == "B":
        return {
            "Grade A (80+) ✅ Achievable": [80, 92]
        }

    return {}

def render_factor_bar(label, val, max_val, color, pts):

    pct = min(abs(val) / max_val * 100, 100)

    pts_str = f"+{pts:.1f}" if pts > 0 else f"{pts:.1f}"

    pts_col = "#4ade80" if pts > 0 else "#f87171"

    st.markdown(f"""
    <div class="factor-row">
        <div class="factor-label">{label}</div>

        <div class="factor-bar-bg">
            <div style="
                width:{pct}%;
                background:{color};
                height:8px;
                border-radius:6px;
                transition:width 0.5s;
            "></div>
        </div>

        <div class="factor-pts" style="color:{pts_col};">
            {pts_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_shap_label(
    feat,
    input_df,
    s_motivation,
    s_parental,
    s_access,
    s_peer,
    s_teacher,
    s_family,
    s_disability,
    s_internet,
    s_extracurricular,
    hours_studied,
    attendance,
    sleep_hours,
    tutoring_sessions,
    physical_activity
):

    """Return friendly label only for features that are actually active/meaningful"""

    # Numeric features — always show
    numeric_map = {
        'Hours_Studied': f'Hours Studied ({hours_studied} hrs/week)',
        'Attendance': f'Attendance ({attendance}%)',
        'Sleep_Hours': f'Sleep Hours ({sleep_hours} hrs/night)',
        'Tutoring_Sessions': f'Tutoring Sessions ({tutoring_sessions}/month)',
        'Physical_Activity': f'Physical Activity ({physical_activity} hrs/week)',
        'Study_Efficiency': 'Study Efficiency',
        'Wellness': 'Wellness Score',
        'Study_Intensity': 'Study Intensity',
        'Score_Consistency': 'Score Consistency',
        'Risk_Factor': 'Risk Factor',
        'Motivation_Adjusted_Study': 'Motivation-Adjusted Study',
        'Attendance_Study_Balance': 'Attendance-Study Balance',
        'Previous_Scores': f'Previous Scores',
    }

    if feat in numeric_map:
        return numeric_map[feat]

    # One-hot features — only show if value == 1 (feature is active)
    val = input_df[feat].iloc[0]

    one_hot_map = {
        'Motivation_Level_Low': (
            'Motivation Level',
            'Low',
            s_motivation == "Low"
        ),

        'Motivation_Level_Medium': (
            'Motivation Level',
            'Medium',
            s_motivation == "Medium"
        ),

        'Parental_Involvement_Low': (
            'Parental Involvement',
            'Low',
            s_parental == "Low"
        ),

        'Parental_Involvement_Medium': (
            'Parental Involvement',
            'Medium',
            s_parental == "Medium"
        ),

        'Access_to_Resources_Low': (
            'Access to Resources',
            'Low',
            s_access == "Low"
        ),

        'Access_to_Resources_Medium': (
            'Access to Resources',
            'Medium',
            s_access == "Medium"
        ),

        'Peer_Influence_Neutral': (
            'Peer Influence',
            'Neutral',
            s_peer == "Neutral"
        ),

        'Peer_Influence_Positive': (
            'Peer Influence',
            'Positive',
            s_peer == "Positive"
        ),

        'Teacher_Quality_Low': (
            'Teacher Quality',
            'Low',
            s_teacher == "Low"
        ),

        'Teacher_Quality_Medium': (
            'Teacher Quality',
            'Medium',
            s_teacher == "Medium"
        ),

        'Family_Income_Low': (
            'Family Income',
            'Low',
            s_family == "Low"
        ),

        'Family_Income_Medium': (
            'Family Income',
            'Medium',
            s_family == "Medium"
        ),

        'Learning_Disabilities_Yes': (
            'Learning Disability',
            'Yes',
            s_disability
        ),

        'Internet_Access_Yes': (
            'Internet Access',
            'Yes',
            s_internet
        ),

        'Extracurricular_Activities_Yes': (
            'Extracurricular',
            'Yes',
            s_extracurricular
        ),

        'Gender_Male': (
            'Gender',
            'Male',
            True
        ),

        'School_Type_Public': (
            'School Type',
            'Public',
            True
        ),

        'Distance_from_Home_Near': (
            'Distance from Home',
            'Near',
            True
        ),

        'Distance_from_Home_Moderate': (
            'Distance from Home',
            'Moderate',
            True
        ),

        'Parental_Education_Level_High School': (
            'Parental Education',
            'High School',
            True
        ),

        'Parental_Education_Level_Postgraduate': (
            'Parental Education',
            'Postgraduate',
            True
        ),
    }

    if feat in one_hot_map:

        label_name, label_val, is_active = one_hot_map[feat]

        if is_active and val == 1:
            return f'{label_name}: {label_val}'

        return None  # Skip inactive one-hot features

    return feat.replace('_', ' ')

# ── Predict Button ───────────────────────────────────────
if st.sidebar.button("🔮 Predict Performance", type="primary"):

    st.session_state.show_plans = False

    try:

        input_df = build_features(
            hours_studied,
            attendance,
            sleep_hours,
            previous_scores,
            tutoring_sessions,
            physical_activity,
            parental_involvement,
            access_to_resources,
            motivation_level,
            family_income,
            teacher_quality,
            peer_influence,
            distance_from_home,
            parental_education,
            extracurricular,
            internet_access,
            learning_disability,
            gender_male
        )

        input_df = input_df[feature_names]

        predicted_score = model.predict(input_df)[0]

        grade, emoji = score_to_grade(predicted_score)

        explainer = shap.TreeExplainer(model)

        shap_vals = explainer.shap_values(input_df)

        st.session_state.predicted_score = predicted_score
        st.session_state.grade = grade
        st.session_state.emoji = emoji
        st.session_state.input_df = input_df
        st.session_state.shap_values = shap_vals[0].tolist()

        st.session_state.s_hours = hours_studied
        st.session_state.s_attendance = attendance
        st.session_state.s_sleep = sleep_hours
        st.session_state.s_tutoring = tutoring_sessions
        st.session_state.s_physical = physical_activity
        st.session_state.s_motivation = motivation_level
        st.session_state.s_parental = parental_involvement
        st.session_state.s_access = access_to_resources
        st.session_state.s_peer = peer_influence
        st.session_state.s_teacher = teacher_quality
        st.session_state.s_family = family_income
        st.session_state.s_disability = learning_disability
        st.session_state.s_internet = internet_access
        st.session_state.s_extracurricular = extracurricular

    except Exception as e:

        st.error(f"❌ Prediction error: {e}")
        st.code(traceback.format_exc())
        st.stop()

# ── Show Results ─────────────────────────────────────────
if st.session_state.predicted_score is not None:

    predicted_score = st.session_state.predicted_score
    grade = st.session_state.grade
    emoji = st.session_state.emoji
    input_df = st.session_state.input_df
    sv = st.session_state.shap_values

    # ── Step 1: Score Cards ──────────────────────────────
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f'''
            <div class="metric-card">
                <div class="metric-value">
                    {predicted_score:.1f}
                    <span style="font-size:1rem;color:#64748b">/100</span>
                </div>
                <div class="metric-label">Predicted Score</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col2:

        grade_colors = {
            "A": "#4ade80",
            "B": "#60a5fa",
            "C": "#facc15",
            "D": "#fb923c",
            "F": "#f87171"
        }

        gc = grade_colors.get(grade, "#e2e8f0")

        st.markdown(
            f'''
            <div class="metric-card">
                <div class="metric-value" style="color:{gc};">
                    {emoji} {grade}
                </div>
                <div class="metric-label">Grade</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col3:

        status = "Pass " if predicted_score >= 50 else "Fail "

        st.markdown(
            f'''
            <div class="metric-card">
                <div class="metric-value">{status}</div>
                <div class="metric-label">Status</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    # ── Step 2: SHAP Factors ─────────────────────────────
    st.markdown(
        '<div class="section-header">📊 What\'s affecting your score?</div>',
        unsafe_allow_html=True
    )

    shap_dict = {
        feature_names[i]: sv[i]
        for i in range(len(feature_names))
    }

    # Build labeled shap pairs — skip inactive one-hot features
    labeled = []

    for feat, val in shap_dict.items():

        label = get_shap_label(
            feat,
            input_df,
            st.session_state.s_motivation,
            st.session_state.s_parental,
            st.session_state.s_access,
            st.session_state.s_peer,
            st.session_state.s_teacher,
            st.session_state.s_family,
            st.session_state.s_disability,
            st.session_state.s_internet,
            st.session_state.s_extracurricular,
            st.session_state.s_hours,
            st.session_state.s_attendance,
            st.session_state.s_sleep,
            st.session_state.s_tutoring,
            st.session_state.s_physical
        )

        if label:
            labeled.append((label, val))

    positive = sorted(
        [(l, v) for l, v in labeled if v > 0],
        key=lambda x: -x[1]
    )[:4]

    negative = sorted(
        [(l, v) for l, v in labeled if v < 0],
        key=lambda x: x[1]
    )[:4]

    max_val = max(
        [abs(v) for _, v in positive + negative] + [0.1]
    )

    col_pos, col_neg = st.columns(2)

    with col_pos:

        st.markdown("**⬆️Pushing your score UP:**")

        for label, val in positive:
            render_factor_bar(label, val, max_val, "#4ade80", val)

    with col_neg:

        st.markdown("**⬇️ Pulling your score DOWN:**")

        for label, val in negative:
            render_factor_bar(label, val, max_val, "#f87171", val)

    # ── Step 3: Grade Target ─────────────────────────────
    if grade != "A":

        st.markdown(
            '<div class="section-header">🎯 What grade do you want to achieve?</div>',
            unsafe_allow_html=True
        )

        grade_options = get_grade_options(grade)

        if grade_options:

            selected_target = st.radio(
                "Choose your target (based on what's realistic for you):",
                list(grade_options.keys()),
                index=0,
                key="grade_radio"
            )

            st.session_state.selected_range = grade_options[selected_target]

            if grade == "C" and "A" in selected_target:

                st.markdown(
                    '''
                    <div class="warning-box">
                    ⚠️ Jumping from C to A is very challenging.
                    The improvement plan may suggest large changes.
                    Consider targeting Grade B first for more achievable steps.
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            if st.button("📋 Generate My Improvement Plans", type="primary"):
                st.session_state.show_plans = True

    else:

        st.success("🎉 You're already at Grade A! Keep it up!")

    # ── Step 4: Plans ────────────────────────────────────
    if st.session_state.show_plans and st.session_state.selected_range:

        desired_range = st.session_state.selected_range

        query = input_df[feature_names].astype(float)

        with st.spinner("Generating your personalized plans..."):

            try:

                full_df = pd.read_csv('processed_student_data.csv')

                train_df = full_df[feature_names].astype(float).copy()

                train_df['Exam_Score'] = full_df['Exam_Score'].values

                dice_data = dice_ml.Data(
                    dataframe=train_df,
                    continuous_features=feature_names,
                    outcome_name='Exam_Score'
                )

                dice_model = dice_ml.Model(
                    model=model,
                    backend='sklearn',
                    model_type='regressor'
                )

                dice_exp = Dice(
                    dice_data,
                    dice_model,
                    method='genetic'
                )

                cf = dice_exp.generate_counterfactuals(
                    query,
                    total_CFs=3,
                    desired_range=desired_range
                )

                cf_df = cf.cf_examples_list[0].final_cfs_df

                actionable = [
                    'Hours_Studied',
                    'Attendance',
                    'Tutoring_Sessions',
                    'Sleep_Hours',
                    'Physical_Activity'
                ]

                disp = {
                    'Hours_Studied': '📚 Hours Studied / week',
                    'Attendance': '🏫 Attendance',
                    'Tutoring_Sessions': '👨‍🏫 Tutoring Sessions / month',
                    'Sleep_Hours': '😴 Sleep Hours / night',
                    'Physical_Activity': '🏃 Physical Activity / week',
                }

                units = {
                    'Hours_Studied': 'hrs',
                    'Attendance': '%',
                    'Tutoring_Sessions': '/mo',
                    'Sleep_Hours': 'hrs',
                    'Physical_Activity': 'hrs',
                }

                plan_names = [
                    "📚 Study Focus",
                    "🏫 Attendance Focus",
                    "⚖️ Balanced Plan"
                ]

                plan_colors = [
                    "#6366f1",
                    "#22d3ee",
                    "#f59e0b"
                ]

                st.markdown(
                    '<div class="section-header">💡 Your 3 Personalized Improvement Plans</div>',
                    unsafe_allow_html=True
                )

                cols = st.columns(3)

                for i, col in enumerate(cols):

                    with col:

                        cf_row = cf_df[feature_names].iloc[i]

                        changes = cf_row - query.iloc[0]

                        pred_score = model.predict(
                            cf_df[feature_names].iloc[[i]]
                        )[0]

                        cf_grade, cf_emoji = score_to_grade(pred_score)

                        score_color = (
                            "#4ade80"
                            if pred_score >= 80
                            else "#60a5fa"
                            if pred_score >= 70
                            else "#facc15"
                        )

                        st.markdown(
                            f'''
                            <div class="plan-card"
                                 style="border-top: 3px solid {plan_colors[i]};">

                                <div class="plan-title"
                                     style="color:{plan_colors[i]};">

                                     {plan_names[i]}

                                </div>
                            ''',
                            unsafe_allow_html=True
                        )

                        shown = 0

                        for feat in actionable:

                            change = changes[feat]

                            if abs(change) > 0.1:

                                curr = query[feat].iloc[0]
                                tgt = curr + change

                                unit = units[feat]
                                name = disp[feat]

                                arrow = "↑" if change > 0 else "↓"

                                arrow_color = (
                                    "#4ade80"
                                    if change > 0
                                    else "#f87171"
                                )

                                st.markdown(
                                    f'''
                                    <div class="plan-item">

                                        <span style="
                                            color:#94a3b8;
                                            font-size:0.78rem;
                                        ">
                                            {name}
                                        </span>

                                        <br>

                                        <b>{curr:.0f}{unit}</b>

                                        <span style="
                                            color:{arrow_color};
                                            font-weight:700;
                                        ">
                                            {arrow}
                                        </span>

                                        <b>{tgt:.0f}{unit}</b>

                                    </div>
                                    ''',
                                    unsafe_allow_html=True
                                )

                                shown += 1

                        if shown == 0:

                            st.markdown(
                                '''
                                <div class="plan-item" style="color:#4ade80;">
                                    ✅ Minor adjustments needed
                                </div>
                                ''',
                                unsafe_allow_html=True
                            )

                        st.markdown(
                            f'''
                            <div class="plan-score"
                                 style="
                                 background:linear-gradient(
                                     135deg,
                                     {plan_colors[i]}22,
                                     {plan_colors[i]}44
                                 );
                                 border:1px solid {plan_colors[i]}66;
                                 ">

                                <span style="color:#94a3b8;">
                                    {predicted_score:.0f}
                                </span>

                                <span style="color:#e2e8f0;">
                                    →
                                </span>

                                <span style="
                                    color:{score_color};
                                    font-size:1.1rem;
                                ">
                                    {pred_score:.0f}
                                </span>

                                <span style="margin-left:6px;">
                                    {cf_emoji} {cf_grade}
                                </span>

                            </div>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )

            except Exception as e:

                st.warning(f"⚠️ Could not generate plans: {e}")

                st.code(traceback.format_exc())

else:

    st.info(
        "👈 Fill in your details in the sidebar and click **Predict Performance**."
    )