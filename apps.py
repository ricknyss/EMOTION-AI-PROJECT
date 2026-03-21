import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# -------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Emotion AI", layout="centered")

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL ---------------------
model = tf.keras.models.load_model("emotion_model.hdf5", compile=False)

# -------------------- LABELS --------------------
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# -------------------- FACE DETECTOR --------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------- UI --------------------
st.title("😊 Face Emotion Detection + Recommendation System")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# -------------------- RECOMMENDATION --------------------
def get_recommendation(emotion):
    if emotion == "Happy":
        return "🎵 [Party Songs](https://www.youtube.com/results?search_query=party+songs)"
    elif emotion == "Sad":
        return "💡 [Motivational Videos](https://www.youtube.com/results?search_query=motivational+videos)"
    elif emotion == "Angry":
        return "🧘 [Meditation Music](https://www.youtube.com/results?search_query=meditation+music)"
    elif emotion == "Fear":
        return "😌 [Calm Music](https://www.youtube.com/results?search_query=calm+music)"
    elif emotion == "Surprise":
        return "🎬 [Trending Videos](https://www.youtube.com/results?search_query=trending+videos)"
    else:
        return "📚 [Study Motivation](https://www.youtube.com/results?search_query=study+motivation)"

# -------------------- HISTORY --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- MAIN LOOP --------------------
while run:
    ret, frame = cap.read()

    if not ret:
        st.error("Camera not working ❌")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        # FIXED SHAPE
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 64, 64, 1))

        prediction = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(prediction)]

        # DRAW BOX()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # SHOW RESULT
        st.subheader(f"Detected Emotion: {emotion}")
        st.markdown(get_recommendation(emotion))

        # STORE HISTORY
        st.session_state.history.append(emotion)

    FRAME_WINDOW.image(frame, channels="BGR")

# -------------------- SHOW HISTORY --------------------
if st.session_state.history:
    st.write("📊 Recent Emotions:", st.session_state.history[-5:])

cap.release()
