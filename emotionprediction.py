import keras
import librosa
import numpy as np
import os
import streamlit as st
import matplotlib.pyplot as plt
import pyttsx3
import requests
from io import BytesIO
from config import EXAMPLES_PATH, MODEL_DIR_PATH
from io import StringIO
import librosa.display

class LivePredictions:
    def __init__(self, file):
        self.file = file
        self.model_path = None
        for model_file in ['Emotion_Voice_Detection_Model.h5', 'Emotion_Voice_Detection_Model.keras']:
            path = os.path.join(MODEL_DIR_PATH, model_file)
            if os.path.exists(path):
                self.model_path = path
                break
        if self.model_path is None:
            raise FileNotFoundError(f"Model file not found in {MODEL_DIR_PATH}")
        self.loaded_model = keras.models.load_model(self.model_path)

    def make_predictions(self):
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=0)  # Add a new dimension for batch size
        x = np.expand_dims(x, axis=-1)     # Add a new dimension for channel
        predictions = self.loaded_model.predict(x)
        predicted_class = np.argmax(predictions, axis=1)
        return predictions, self.convert_class_to_emotion(predicted_class)

    @staticmethod
    def convert_class_to_emotion(pred):
        label_conversion = {
            '0': 'neutral',
            '1': 'calm',
            '2': 'happy',
            '3': 'sad',
            '4': 'angry',
            '5': 'fearful',
            '6': 'disgust',
            '7': 'surprised'
        }
        label = label_conversion.get(str(pred[0]), "Unknown emotion")
        return label

    def get_model_summary(self):
        stream = StringIO()
        self.loaded_model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()
        stream.close()
        return summary_str

    def plot_waveform(self):
        data, sampling_rate = librosa.load(self.file)
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, len(data) / sampling_rate, len(data)), data)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        st.pyplot(plt)
        plt.close()

    def plot_mfcc(self):
        data, sampling_rate = librosa.load(self.file)
        mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

    def plot_prediction_probabilities(self, predictions):
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        plt.figure(figsize=(8, 4))
        plt.bar(emotions, predictions[0], color='lightblue')
        plt.title('Prediction Probabilities')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        plt.close()

    def generate_emotional_voice(self, emotion):
        """Generate voice output based on predicted emotion."""
        engine = pyttsx3.init()
        if emotion == 'happy':
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
        elif emotion == 'sad':
            engine.setProperty('rate', 90)
            engine.setProperty('volume', 0.7)
        elif emotion == 'angry':
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
        elif emotion in ['fearful', 'disgust']:
            engine.setProperty('rate', 100)
            engine.setProperty('volume', 0.5)
        else:
            engine.setProperty('rate', 120)
            engine.setProperty('volume', 1.0)

        text = f"I am feeling {emotion}."
        engine.say(text)
        engine.runAndWait()

# Streamlit App
def main():
    st.title("Emotion Voice Detection")
    st.write("Upload an audio file to predict the emotion and view audio-related graphs.")

    # Upload file
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        with open(os.path.join(EXAMPLES_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create an instance of LivePredictions
        live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, uploaded_file.name))
        
        # Display the model summary
        st.write("### Model Summary")
        model_summary = live_prediction.get_model_summary()
        st.text(model_summary)

        # Display Waveform
        st.write("### Waveform")
        live_prediction.plot_waveform()

        # Display MFCCs
        st.write("### MFCC (Mel-Frequency Cepstral Coefficients)")
        live_prediction.plot_mfcc()

        # Make predictions and plot probabilities
        st.write("### Prediction")
        predictions, predicted_emotion = live_prediction.make_predictions()
        st.success(f"The predicted emotion is: {predicted_emotion}")

        # Display prediction probabilities
        st.write("### Prediction Probabilities")
        live_prediction.plot_prediction_probabilities(predictions)

        # Generate emotional voice
        st.write("### Generating Emotional Voice")
        live_prediction.generate_emotional_voice(predicted_emotion)
        st.success("Emotional voice generated!")

if __name__ == '__main__':
    main()
