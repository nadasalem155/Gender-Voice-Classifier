import streamlit as st
st.set_page_config(page_title="🎤 Voice Gender Recognition", page_icon="🗣️", layout="centered")

import tempfile
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

# --- Load Keras model once ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# --- Preprocessing function ---
def preprocess_audio(filename, max_len=48000):
    wav, sr = librosa.load(filename, sr=16000, mono=True)
    if len(wav) > max_len:
        wav = wav[:max_len]
    else:
        wav = np.pad(wav, (0, max_len - len(wav)))
    
    spec = np.abs(librosa.stft(wav, n_fft=512, hop_length=256))
    spec = np.expand_dims(spec, -1)
    spec = tf.image.resize(spec, [128, 128])
    spec = np.expand_dims(spec, 0)
    return spec, wav, sr

# --- Prediction function ---
def predict_gender(file_path):
    features, _, _ = preprocess_audio(file_path)
    pred = model.predict(features)
    return "👨‍🦱 Male" if pred[0][0] > 0.5 else "👩‍🦰 Female"

# --- Initialize session state ---
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Streamlit UI ---
st.title("🎤 Voice Gender Recognition")
st.markdown("Detect whether a voice belongs to a **Male** or **Female** using a CNN model.")

# --- Upload audio file section ---
st.subheader("📂 Upload an Audio File")
uploaded_file = st.file_uploader("Choose a file (wav, mp3, ogg) 🎧", type=["wav","mp3","ogg"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        st.session_state.uploaded_path = tmp_file.name
    
    st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)

# Display uploaded file result
if st.session_state.uploaded_path and st.session_state.uploaded_result:
    st.success(f"Prediction (Uploaded): {st.session_state.uploaded_result}")
    
    _, wav, sr = preprocess_audio(st.session_state.uploaded_path)
    plt.figure(figsize=(8,2))
    plt.title("📈 Waveform")
    plt.plot(wav, color="#1f77b4")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    st.pyplot(plt)
    
    st.audio(st.session_state.uploaded_path, format="audio/wav")

# --- Remove button for uploaded file ---
if st.button("🗑️ Remove Uploaded File"):
    st.session_state.uploaded_path = None
    st.session_state.uploaded_result = None
    st.experimental_rerun()

# --- Record audio section ---
st.subheader("🎤 Record Your Voice")
st.markdown("Press **Record** to capture your voice from the browser 🎙️.")
audio_bytes = audio_recorder()
if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        st.session_state.recorded_path = tmp_file.name
    
    st.session_state.recorded_result = predict_gender(st.session_state.recorded_path)

# Display recorded audio result
if st.session_state.recorded_path and st.session_state.recorded_result:
    st.success(f"Prediction (Recorded): {st.session_state.recorded_result}")
    
    _, wav, sr = preprocess_audio(st.session_state.recorded_path)
    plt.figure(figsize=(8,2))
    plt.title("📈 Waveform")
    plt.plot(wav, color="#ff7f0e")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    st.pyplot(plt)
    
    st.audio(st.session_state.recorded_path, format="audio/wav")

# --- Remove button for recorded audio ---
if st.button("🗑️ Remove Recording"):
    st.session_state.recorded_path = None
    st.session_state.recorded_result = None
    st.experimental_rerun()
