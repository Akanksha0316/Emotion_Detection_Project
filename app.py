import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Emotion Detection from Text",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* =====================================================
       MAIN APPLICATION
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(124, 92, 255, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(168, 85, 247, 0.14),
                transparent 30%
            ),
            #09090f;
        color: #f5f5f7;
    }


    /* Main content width */

    /* Hide Streamlit's top header and toolbar */
[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Control the top spacing of the application */
.block-container {
    max-width: 1150px;
    padding-top: 0.8rem !important;
    padding-bottom: 3rem;
}


    /* =====================================================
       HERO SECTION
       ===================================================== */

    .hero {
        background:
            linear-gradient(
                135deg,
                #15121f,
                #0e0d15
            );
        border: 1px solid #30294a;
        border-radius: 24px;
        padding: 42px 44px;
        margin-bottom: 32px;

        box-shadow:
            0 15px 45px rgba(0, 0, 0, 0.45),
            0 0 40px rgba(124, 92, 255, 0.08);
    }


    .hero-title {
        font-size: 44px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 12px;
    }


    .hero-subtitle {
        font-size: 17px;
        color: #b7b3c7;
        line-height: 1.7;
        max-width: 850px;
    }


    .badge {
        display: inline-block;

        background:
            linear-gradient(
                135deg,
                #241b48,
                #321d54
            );

        color: #c8b8ff;

        border: 1px solid #59458f;

        padding: 7px 15px;

        border-radius: 50px;

        font-size: 13px;

        font-weight: 700;

        margin-bottom: 16px;
    }


    /* =====================================================
       SECTION HEADINGS
       ===================================================== */

    .section-title {
        font-size: 26px;
        font-weight: 750;
        color: #ffffff;
        margin-top: 12px;
        margin-bottom: 8px;
    }


    .section-description {
        color: #a7a3b5;
        font-size: 15px;
        margin-bottom: 18px;
    }


    /* =====================================================
       TEXT INPUT
       ===================================================== */

    textarea {
        background-color: #12111a !important;

        color: #f5f5f7 !important;

        border: 1px solid #39334e !important;

        border-radius: 16px !important;

        font-size: 16px !important;
    }


    textarea::placeholder {
        color: #777285 !important;
    }


    textarea:focus {
        border-color: #8b6cff !important;

        box-shadow:
            0 0 0 2px rgba(139,108,255,0.18) !important;
    }


    /* =====================================================
       PREDICT BUTTON
       ===================================================== */

    div.stButton > button {

        width: 100%;

        border-radius: 14px;

        height: 54px;

        font-size: 16px;

        font-weight: 700;

        border: 1px solid #8b6cff;

        background:
            linear-gradient(
                135deg,
                #6d4aff,
                #9b5cff
            );

        color: white;

        transition: all 0.25s ease;

        box-shadow:
            0 8px 25px rgba(109, 74, 255, 0.20);
    }


    div.stButton > button:hover {

        transform: translateY(-2px);

        border-color: #b09aff;

        box-shadow:
            0 12px 30px rgba(109, 74, 255, 0.35);
    }


    /* =====================================================
       RESULT CARD
       ===================================================== */

    .result-card {

        background:
            linear-gradient(
                145deg,
                #171426,
                #0f0e16
            );

        border: 1px solid #44376c;

        border-radius: 22px;

        padding: 32px;

        margin-top: 25px;

        box-shadow:
            0 12px 40px rgba(0,0,0,0.40),
            0 0 35px rgba(124,92,255,0.08);

        text-align: center;
    }


    .result-label {

        color: #a7a0bb;

        font-size: 14px;

        font-weight: 600;

        text-transform: uppercase;

        letter-spacing: 1.5px;
    }


    .emotion-result {

        font-size: 44px;

        font-weight: 850;

        color: #b49cff;

        margin-top: 7px;

        margin-bottom: 8px;

        text-shadow:
            0 0 20px rgba(180,156,255,0.20);
    }


    .confidence-result {

        font-size: 20px;

        color: #e7e3f0;

        font-weight: 700;
    }


    /* =====================================================
       INFO CARDS
       ===================================================== */

    .info-card {

        background:
            linear-gradient(
                145deg,
                #15131e,
                #0f0e15
            );

        border: 1px solid #302b42;

        border-radius: 18px;

        padding: 23px;

        height: 100%;

        box-shadow:
            0 7px 25px rgba(0,0,0,0.25);

        transition: all 0.2s ease;
    }


    .info-card:hover {

        border-color: #584a83;

        transform: translateY(-2px);

        box-shadow:
            0 10px 30px rgba(0,0,0,0.35);
    }


    .info-icon {

        font-size: 29px;

        margin-bottom: 9px;
    }


    .info-title {

        font-size: 18px;

        font-weight: 750;

        color: #ffffff;

        margin-bottom: 7px;
    }


    .info-text {

        color: #aaa5b8;

        font-size: 14px;

        line-height: 1.6;
    }


    /* =====================================================
       EMOTION CHIPS
       ===================================================== */

    .emotion-chip {

        display: inline-block;

        background:
            linear-gradient(
                135deg,
                #1b1730,
                #24193a
            );

        color: #c8b9ff;

        border: 1px solid #433568;

        border-radius: 50px;

        padding: 9px 16px;

        margin: 5px;

        font-size: 14px;

        font-weight: 650;

        transition: all 0.2s ease;
    }


    .emotion-chip:hover {

        background: #302052;

        border-color: #7157ad;

        transform: translateY(-1px);
    }


    /* =====================================================
       STREAMLIT PROGRESS BAR
       ===================================================== */

    .stProgress > div > div > div > div {

        background:
            linear-gradient(
                90deg,
                #6d4aff,
                #b05cff
            );
    }


    /* =====================================================
       DIVIDERS
       ===================================================== */

    hr {

        border-color: #2a2734 !important;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {

        text-align: center;

        color: #777285;

        font-size: 13px;

        margin-top: 45px;

        padding-top: 22px;

        border-top: 1px solid #292631;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "model/emotion_detection_bilstm_fixed.keras"
    )


# =========================================================
# LOAD TOKENIZER
# =========================================================

@st.cache_resource
def load_tokenizer():

    with open("model/tokenizer.pkl", "rb") as file:
        return pickle.load(file)


# =========================================================
# LOAD EMOTION LABELS
# =========================================================

@st.cache_resource
def load_emotion_labels():

    with open("model/emotion_labels.pkl", "rb") as file:
        return pickle.load(file)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_emotion(
    text,
    model,
    tokenizer,
    emotion_labels
):

    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    max_length = model.input_shape[1]

    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_label = prediction.argmax(
        axis=1
    )[0]

    emotion = emotion_labels[predicted_label]

    confidence = (
        prediction[0][predicted_label] * 100
    )

    return emotion, confidence, prediction[0]


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    model = load_model()
    tokenizer = load_tokenizer()
    emotion_labels = load_emotion_labels()

except Exception as e:

    st.error(
        "Unable to load the model or supporting files."
    )

    st.exception(e)

    st.stop()


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

    <div class="badge">DEEP LEARNING • NLP</div>

    <div class="hero-title">
        🧠 Emotion Detection from Text
    </div>

    <div class="hero-subtitle">
        Discover the emotion expressed in your text using
        a trained Bidirectional LSTM deep learning model.
        Enter any sentence and get an instant prediction
        with a confidence score.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">✍️ Analyze Your Text</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Write a sentence below and let the AI model analyze '
    'the emotion expressed in it.'
    '</div>',
    unsafe_allow_html=True
)


text = st.text_area(
    "Your text",
    placeholder=(
        "Example: I am extremely happy today "
        "because I achieved my goal!"
    ),
    height=160,
    label_visibility="collapsed"
)


# =========================================================
# PREDICT BUTTON
# =========================================================

predict_button = st.button(
    "🔍  Predict Emotion",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION RESULT
# =========================================================

if predict_button:

    if not text.strip():

        st.warning(
            "⚠️ Please enter some text before predicting."
        )

    else:

        with st.spinner("Analyzing your text..."):

            emotion, confidence, probabilities = (
                predict_emotion(
                    text,
                    model,
                    tokenizer,
                    emotion_labels
                )
            )

        emotion_icons = {
            "sadness": "😢",
            "joy": "😊",
            "love": "❤️",
            "anger": "😠",
            "fear": "😨",
            "surprise": "😮"
        }

        icon = emotion_icons.get(
            emotion.lower(),
            "🧠"
        )

        # Result card
        st.markdown(
            f"""
        <div class="result-card">
            <div class="result-label">Detected Emotion</div>
            <div class="emotion-result">{icon} {emotion.upper()}</div>
            <div class="confidence-result">
                Confidence: {float(confidence):.2f}%
            </div>
        </div>
        """,
            unsafe_allow_html=True
        )


        # Confidence progress
        st.write("")

        st.markdown(
            "**Prediction Confidence**"
        )

        st.progress(
        min(float(confidence) / 100.0, 1.0)
        )


        # =================================================
        # PROBABILITY BREAKDOWN
        # =================================================

        st.write("")

        st.markdown(
            "### 📊 Emotion Probability"
        )

        emotion_names = [
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise"
        ]

        for i, emotion_name in enumerate(emotion_names):

            probability = float(
                probabilities[i]
            )

            st.write(
                f"**{emotion_name.capitalize()}** "
                f"— {probability * 100:.2f}%"
            )

            st.progress(
              min(float(probability), 1.0)
            )


# =========================================================
# EMOTION CLASSES
# =========================================================

st.write("")

st.markdown(
    '<div class="section-title">🎭 Supported Emotions</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div style="margin-top: 10px; margin-bottom: 25px;">

<span class="emotion-chip">😢 Sadness</span>
<span class="emotion-chip">😊 Joy</span>
<span class="emotion-chip">❤️ Love</span>
<span class="emotion-chip">😠 Anger</span>
<span class="emotion-chip">😨 Fear</span>
<span class="emotion-chip">😮 Surprise</span>

</div>
""", unsafe_allow_html=True)


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'The application processes your text through several '
    'steps before producing the final prediction.'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
<div class="info-card">
    <div class="info-icon">📝</div>
    <div class="info-title">1. Text Input</div>
    <div class="info-text">
        Enter a sentence or any short piece of text
        that you want the model to analyze.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
<div class="info-card">
    <div class="info-icon">🧹</div>
    <div class="info-title">2. Processing</div>
    <div class="info-text">
        The text is cleaned, tokenized and converted
        into a numerical sequence that the neural
        network can understand.
    </div>
</div>
""",
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
<div class="info-card">
    <div class="info-icon">🤖</div>
    <div class="info-title">3. Prediction</div>
    <div class="info-text">
        The trained BiLSTM model analyzes the sequence
        and predicts the most likely emotion.
    </div>
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.write("")

st.markdown(
    '<div class="section-title">🧪 Model Information</div>',
    unsafe_allow_html=True
)

info1, info2, info3, info4 = st.columns(4)


with info1:

    st.markdown(
        """
<div class="info-card">
    <div class="info-title">Architecture</div>
    <div class="info-text">Bidirectional LSTM</div>
</div>
""",
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
<div class="info-card">
    <div class="info-title">Task</div>
    <div class="info-text">Text Emotion Classification</div>
</div>
""",
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
<div class="info-card">
    <div class="info-title">Classes</div>
    <div class="info-text">6 Emotion Categories</div>
</div>
""",
        unsafe_allow_html=True
    )


with info4:

    st.markdown(
        """
<div class="info-card">
    <div class="info-title">Technology</div>
    <div class="info-text">Python • TensorFlow • Streamlit</div>
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    Emotion Detection from Text •
    Deep Learning & Natural Language Processing

</div>
""", unsafe_allow_html=True)