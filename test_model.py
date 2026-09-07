import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------

print("Loading model...")

model = tf.keras.models.load_model(
    "model/emotion_detection_bilstm_fixed.keras"
)

print("Model loaded successfully!")


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

with open("model/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

print("Tokenizer loaded successfully!")


# --------------------------------------------------
# 3. Load emotion labels
# --------------------------------------------------

print("Loading emotion labels...")

with open("model/emotion_labels.pkl", "rb") as file:
    emotion_labels = pickle.load(file)

print("Emotion labels loaded successfully!")


# --------------------------------------------------
# 4. Text cleaning function
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# 5. Prediction function
# --------------------------------------------------

def predict_emotion(text):

    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

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

    predicted_label = prediction.argmax(axis=1)[0]

    emotion = emotion_labels[predicted_label]

    confidence = prediction[0][predicted_label] * 100

    return emotion, confidence


# --------------------------------------------------
# 6. Test the model
# --------------------------------------------------

print()
print("========================================")
print("EMOTION DETECTION TEST")
print("========================================")

text = input("Enter a sentence: ")

emotion, confidence = predict_emotion(text)

print()
print("Input:", text)
print("Predicted Emotion:", emotion)
print("Confidence:", f"{confidence:.2f}%")