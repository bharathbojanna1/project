import streamlit as st
import os
import whisper
import speech_recognition as sr
import threading
import queue
import tempfile
from PIL import Image
from utils.img_pr import predict_skin_disease
from utils.deepseek_api import ask_deepseek

# Load Whisper model (for real-time transcription)
whisper_model = whisper.load_model("base")

# Page Configuration
st.set_page_config(
    page_title="VetAssist - Pet Health Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem !important; color: #2C3E50; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.5rem !important; color: #3498DB; margin-top: 1rem; margin-bottom: 1rem; }
    .result-container { background-color: #F8F9FA; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; }
    .footer { text-align: center; margin-top: 3rem; color: #7F8C8D; font-size: 0.8rem; }
    .alert-warning { color: #D35400; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Select an Option:", ["Chatbox", "Checkup", "Skin Disease Analyzer"])

# Main Header
st.markdown("<h1 class='main-header'>🐶🐱 VetAssist: Pet Health Assistant</h1>", unsafe_allow_html=True)

# Queue to handle real-time transcription updates
transcription_queue = queue.Queue()

# Function for real-time speech-to-text
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.info("🎙 Listening... Speak now!")
        while True:
            try:
                audio = recognizer.listen(source, timeout=5)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio.write(audio.get_wav_data())
                    temp_audio_path = temp_audio.name

                transcription = whisper_model.transcribe(temp_audio_path)["text"]
                transcription_queue.put(transcription)
            except sr.WaitTimeoutError:
                transcription_queue.put("[No speech detected, try again.]")
            except Exception as e:
                transcription_queue.put(f"[Error: {str(e)}]")
                break

# Chatbox with AI Response & Live Speech-to-Text
if page == "Chatbox":
    st.markdown("<h2 class='section-header'>💬 Ask a Veterinary Question</h2>", unsafe_allow_html=True)

    # Live transcription handler
    if st.button("🎙 Start Voice Input"):
        st.session_state["transcribed_text"] = ""
        threading.Thread(target=transcribe_speech, daemon=True).start()

    # Fetch latest transcription update
    if "transcribed_text" not in st.session_state:
        st.session_state["transcribed_text"] = ""

    while not transcription_queue.empty():
        latest_text = transcription_queue.get()
        st.session_state["transcribed_text"] = latest_text

    # Display real-time transcribed text
    user_query = st.text_area("Type your question here:", value=st.session_state["transcribed_text"], height=100)

    if st.button("Submit Question"):
        if user_query:
            with st.spinner("Fetching AI response..."):
                answer = ask_deepseek(user_query)
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.markdown("### 🤖 AI Response")
                st.markdown(answer)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a question first.")

# Checkup Form
elif page == "Checkup":
    st.markdown("<h2 class='section-header'>🩺 Pet Health Checkup</h2>", unsafe_allow_html=True)

    with st.form("checkup_form"):
        pet_name = st.text_input("Pet's Name")
        pet_breed = st.text_input("Breed")
        pet_age = st.number_input("Age (in years)", min_value=0.1, step=0.1)
        pet_sex = st.selectbox("Sex", ["Male", "Female"])
        pet_weight = st.number_input("Weight (kg)", min_value=0.1, step=0.1)
        pet_symptoms = st.text_area("Describe the symptoms")

        submitted = st.form_submit_button("Get Diagnosis")

        if submitted:
            if pet_symptoms:
                with st.spinner("Generating diagnosis..."):
                    llm_prompt = f"""
                    A {pet_age}-year-old {pet_sex.lower()} {pet_breed} named {pet_name} weighing {pet_weight}kg is experiencing the following symptoms: {pet_symptoms}.
                    What could be the possible health issue and recommendations?
                    """
                    diagnosis = ask_deepseek(llm_prompt)

                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    st.markdown("### 🩺 AI Diagnosis")
                    st.markdown(diagnosis)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Please provide symptoms for analysis.")

# Skin Disease Analyzer
elif page == "Skin Disease Analyzer":
    st.markdown("<h2 class='section-header'>📷 Upload Image for Skin Disease Detection</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a clear image of the pet's skin condition", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            try:
                disease_prediction, confidence = predict_skin_disease(image)

                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.markdown("### 🩺 Diagnosis Results")
                st.markdown(f"**Predicted Condition:** {disease_prediction}")
                st.markdown(f"**Confidence:** {confidence:.2%}")

                if confidence < 0.40:
                    st.markdown("<p class='alert-warning'>⚠️ Prediction Not Accurate. Please consult a veterinarian.</p>", unsafe_allow_html=True)
                else:
                    with st.spinner("Fetching treatment recommendations..."):
                        deepseek_recommendations = ask_deepseek(f"What are the best treatments for {disease_prediction} in pets?")
                        st.markdown("### 📋 Treatment Recommendations")
                        st.markdown(deepseek_recommendations)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# Footer
st.markdown("<div class='footer'>VetAssist is an AI-powered tool and should not replace professional veterinary care.</div>", unsafe_allow_html=True)
