
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="PulseTune",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(120, 90, 255, 0.10), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(255, 90, 160, 0.08), transparent 28%),
        linear-gradient(135deg, #f8f7ff 0%, #ffffff 48%, #f7f8ff 100%);
}

/* Main container */
.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    text-align: center;
    padding: 1rem 0 2.2rem 0;
}

.logo {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 70px;
    height: 70px;
    border-radius: 22px;
    background: linear-gradient(135deg, #6c5ce7, #a855f7);
    color: white;
    font-size: 34px;
    box-shadow: 0 12px 30px rgba(108, 92, 231, 0.25);
    margin-bottom: 15px;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1.8px;
    margin: 0;
    background: linear-gradient(90deg, #5b4bc4, #9b4de8, #d14f9a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #73758a;
    font-size: 1.05rem;
    margin-top: 8px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(220,220,235,0.85);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 10px 35px rgba(50,45,90,0.07);
    backdrop-filter: blur(10px);
    margin-bottom: 22px;
}

.card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #24243a;
    margin-bottom: 18px;
}

.card-subtitle {
    color: #85869a;
    font-size: 0.9rem;
    margin-top: -12px;
    margin-bottom: 20px;
}

/* Inputs */
.stSelectbox label,
.stNumberInput label {
    font-weight: 600 !important;
    color: #35354c !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div {
    border-radius: 12px !important;
}

/* Button */
.stButton > button {
    border: none !important;
    border-radius: 13px !important;
    height: 50px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #6655d9, #9656db) !important;
    color: white !important;
    box-shadow: 0 8px 20px rgba(102,85,217,0.25);
    transition: 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(102,85,217,0.32);
}

/* Metrics */
[data-testid="stMetric"] {
    background: #fafaff;
    border: 1px solid #ededf7;
    padding: 17px;
    border-radius: 16px;
}

[data-testid="stMetricLabel"] {
    font-weight: 600;
}

/* Recommendation card */
.song-card {
    background: rgba(255,255,255,0.94);
    border: 1px solid #e9e8f2;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 25px rgba(45,40,80,0.06);
}

.rank {
    color: #806ee0;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.song-title {
    color: #222236;
    font-size: 1.35rem;
    font-weight: 750;
    margin-top: 5px;
}

.artist {
    color: #77788c;
    font-size: 0.92rem;
    margin-bottom: 12px;
}

.badge {
    display: inline-block;
    background: #f1efff;
    color: #6555c5;
    padding: 6px 12px;
    border-radius: 30px;
    font-size: 0.78rem;
    font-weight: 700;
}

/* Info cards */
.info-box {
    background: #fafaff;
    border: 1px solid #ededf6;
    border-radius: 16px;
    padding: 18px;
    min-height: 120px;
}

.info-number {
    font-size: 1.4rem;
    font-weight: 800;
    color: #705fd1;
}

.info-title {
    font-weight: 700;
    color: #303044;
    margin: 5px 0;
}

.info-text {
    color: #858598;
    font-size: 0.82rem;
    line-height: 1.5;
}

/* Footer */
.footer {
    text-align: center;
    color: #9292a3;
    font-size: 0.78rem;
    margin-top: 45px;
    padding-top: 22px;
    border-top: 1px solid #e8e8f0;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="logo">🎵</div>
    <h1 class="hero-title">PulseTune</h1>
    <div class="hero-subtitle">
        Personalized music recommendations powered by mood & heart rate
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# LOAD DATA
# =========================
try:
    df = pd.read_csv("music_dataset_processed.csv")
except FileNotFoundError:
    st.error("music_dataset_processed.csv not found.")
    st.stop()


# =========================
# LOAD MODELS
# =========================
try:
    valence_model = joblib.load("model/valence_model.pkl")
    arousal_model = joblib.load("model/arousal_model.pkl")
    energy_model = joblib.load("model/energy_model.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please check the model folder.")
    st.stop()


# =========================
# MOOD MAPPING
# =========================
mood_mapping = {
    "Angry/Tense": 0,
    "Calm/Relaxed": 1,
    "Happy/Energetic": 2,
    "Sad/Depressed": 3
}


# =========================
# USER INPUT
# =========================
st.markdown("""
<div class="card">
    <div class="card-title">🎧 Personalize Your Listening</div>
    <div class="card-subtitle">
        Enter your current mood and heart rate to discover music that matches you.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    user_mood = st.selectbox(
        "😊 Current Mood",
        list(mood_mapping.keys())
    )

with col2:
    user_hr = st.number_input(
        "❤️ Heart Rate (BPM)",
        min_value=60,
        max_value=120,
        value=98,
        step=1
    )

st.write("")

_, button_col, _ = st.columns([1, 1.4, 1])

with button_col:
    recommend_clicked = st.button(
        "✨ Find My Music",
        use_container_width=True
    )


# =========================
# RECOMMENDATION
# =========================
if recommend_clicked:

    user_mood_encoded = mood_mapping[user_mood]

    user_input = pd.DataFrame({
        "mood_encoded": [user_mood_encoded],
        "HR_simulated": [user_hr]
    })

    # Predictions
    predicted_valence = valence_model.predict(user_input)[0]
    predicted_arousal = arousal_model.predict(user_input)[0]
    predicted_energy = energy_model.predict(user_input)[0]

    # =========================
    # PREDICTION SECTION
    # =========================
    st.markdown("")
    st.markdown("""
    <div class="card">
        <div class="card-title">🤖 Music Profile</div>
        <div class="card-subtitle">
            Predicted characteristics used for personalized song matching.
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "🎼 Valence",
            f"{predicted_valence:.3f}"
        )

    with m2:
        st.metric(
            "⚡ Arousal",
            f"{predicted_arousal:.3f}"
        )

    with m3:
        st.metric(
            "🔊 Energy",
            f"{predicted_energy:.3f}"
        )

    # =========================
    # FILTER BY MOOD
    # =========================
    filtered_df = df[df["mood"] == user_mood].copy()

    # =========================
    # AUDIO CHECK
    # =========================
    def audio_exists(music_id):
        return os.path.exists(
            os.path.join("audio", f"{music_id}.mp3")
        )

    filtered_df["audio_available"] = filtered_df[
        "musicId"
    ].apply(audio_exists)

    playable_df = filtered_df[
        filtered_df["audio_available"] == True
    ].copy()

    if len(playable_df) < 3:

        st.warning(
            f"Only {len(playable_df)} playable song(s) are available "
            f"for {user_mood}."
        )

        st.info(
            "Add more MP3 files to the audio folder to generate "
            "three playable recommendations."
        )

        st.stop()

    # =========================
    # STANDARD DEVIATIONS
    # =========================
    valence_std = playable_df["Valence(mean)"].std()
    arousal_std = playable_df["Arousal(mean)"].std()
    energy_std = playable_df[
        "pcm_RMSenergy_sma_amean"
    ].std()

    if valence_std == 0:
        valence_std = 1

    if arousal_std == 0:
        arousal_std = 1

    if energy_std == 0:
        energy_std = 1

    # =========================
    # MATCHING DISTANCE
    # =========================
    playable_df["distance"] = np.sqrt(

        (
            (
                playable_df["Valence(mean)"]
                - predicted_valence
            ) / valence_std
        ) ** 2

        +

        (
            (
                playable_df["Arousal(mean)"]
                - predicted_arousal
            ) / arousal_std
        ) ** 2

        +

        (
            (
                playable_df["pcm_RMSenergy_sma_amean"]
                - predicted_energy
            ) / energy_std
        ) ** 2
    )

    top_3 = (
        playable_df
        .sort_values("distance")
        .head(3)
    )

    # =========================
    # RECOMMENDATIONS
    # =========================
    st.markdown("")
    st.markdown("""
    <div class="card">
        <div class="card-title">🎶 Your Music Recommendations</div>
        <div class="card-subtitle">
            Curated from songs matching your selected mood and predicted music profile.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success(
        f"✨ 3 songs found for {user_mood} • {user_hr} BPM"
    )

    for index, (_, row) in enumerate(
        top_3.iterrows(),
        start=1
    ):

        st.markdown(f"""
        <div class="song-card">
            <div class="rank">Recommendation {index}</div>
            <div class="song-title">🎵 {row["title"]}</div>
            <div class="artist">by {row["artist"]}</div>
            <span class="badge">♪ {row["mood"]}</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.caption("VALENCE")
            st.write(
                f"{row['Valence(mean)']:.3f}"
            )

        with c2:
            st.caption("AROUSAL")
            st.write(
                f"{row['Arousal(mean)']:.3f}"
            )

        with c3:
            st.caption("ENERGY")
            st.write(
                f"{row['pcm_RMSenergy_sma_amean']:.3f}"
            )

        audio_path = os.path.join(
            "audio",
            f"{row['musicId']}.mp3"
        )

        st.audio(
            audio_path,
            format="audio/mp3"
        )

        st.markdown("<br>", unsafe_allow_html=True)


# =========================
# HOW IT WORKS
# =========================
st.markdown("")
st.markdown("""
<div class="card">
    <div class="card-title">⚙️ How PulseTune Works</div>
    <div class="card-subtitle">
        A simple data-driven pipeline for personalized music discovery.
    </div>
</div>
""", unsafe_allow_html=True)

i1, i2, i3, i4 = st.columns(4)

with i1:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">01</div>
        <div class="info-title">User Input</div>
        <div class="info-text">
            Mood and heart-rate information are provided by the user.
        </div>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">02</div>
        <div class="info-title">Prediction</div>
        <div class="info-text">
            Random Forest predicts suitable music characteristics.
        </div>
    </div>
    """, unsafe_allow_html=True)

with i3:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">03</div>
        <div class="info-title">Smart Matching</div>
        <div class="info-text">
            Songs are filtered by mood and matched using music features.
        </div>
    </div>
    """, unsafe_allow_html=True)

with i4:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">04</div>
        <div class="info-title">Top 3</div>
        <div class="info-text">
            The closest playable songs are presented with audio playback.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    <b>PulseTune</b> · Pulse-Based Personalized Music Recommendation System
    <br>
    Python · PySpark · Random Forest · Streamlit
</div>
""", unsafe_allow_html=True)

