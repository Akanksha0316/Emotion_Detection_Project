# Emotion Detection from Text

An AI-based emotion detection system that predicts the emotion
expressed in a given text using Deep Learning and Natural Language Processing.

## Project Overview

This project uses a Bidirectional LSTM (BiLSTM) neural network
to classify text into six different emotions.

## Emotions Detected

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

## Technologies Used

- Python
- TensorFlow
- Keras
- NLP
- Bidirectional LSTM
- Streamlit
- NumPy
- Pickle

## Dataset

The model was trained using the Emotion dataset containing
six emotion categories.

## Model

The project uses a trained Bidirectional LSTM model for
emotion classification.

## Project Structure

```text
Emotion_Detection_Project/
│
├── model/
│   ├── emotion_detection_bilstm_fixed.keras
│   ├── emotion_labels.pkl
│   └── tokenizer.pkl
│
├── app.py
├── test_model.py
├── fix_model.py
├── Emotion_Detection_From_Text.ipynb
├── README.md
└── .gitignore
