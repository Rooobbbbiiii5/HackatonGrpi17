import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import time
import av
from dotenv import load_dotenv
import os
import google.genai as genai
from keras.models import load_model
import cv2
import numpy as np
MODEL_PATH = r"C:\Users\fluff\Desktop\buildingblocs\hackathon_project\keras_model_fixed.h5"
LABELS_PATH = r"C:\Users\fluff\Desktop\buildingblocs\hackathon_project\labels.txt"
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
sign_directory = r"C:\Users\fluff\Desktop\buildingblocs\hackathon_project\sign_images"
try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

try:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    if not class_names:
        st.error("No class names found in labels.txt.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load labels: {e}")
    st.stop()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        # If your model was NOT trained on flipped images, comment out the next line
        img_flipped = cv2.flip(img_resized, 1)
        img_norm = (np.asarray(img_flipped, dtype=np.float32).reshape(1, 224, 224, 3) / 127.5) - 1
        try:
            prediction = model.predict(img_norm)
            index = int(np.argmax(prediction))
            class_name = class_names[index]
            confidence_score = float(prediction[0][index])
            print(f"Prediction: {prediction}, Index: {index}, Class: {class_name}, Confidence: {confidence_score}")
        except Exception as e:
            class_name = "Error"
            confidence_score = 0.0
            print(f"Prediction error: {e}")
        img_with_border = cv2.copyMakeBorder(
            img_flipped, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 255, 0)
        )
        cv2.putText(
            img_with_border,
            f"{class_name}: {confidence_score*100:.1f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(img_with_border, format="bgr24")

signlist = []
image_list = []
if "translating" not in st.session_state:
    st.session_state.translating = False
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False


def translate_callback():
    if st.session_state.to_be_translated.strip():
        st.session_state.translating = True

def home():
    st.title("GROUP INTERMEDIATE -- 17")
    st.divider()
    st.title("Hackathon Project: Sign Language Recognition and Generation")
    st.markdown("There are often many challenges when it comes to communication between modern English speakers and the hearing-impaired community. However, despite this being a major challenge in the world, there aren't many people who are aware of this problem. This application is our solution to this problem. With it able to serve as a sign language translator, it enables two-way communication between English speakers and the hearing-impaired community. It translates English into sign language and vice versa. By having translation between the two communities, this application helps to promote inclusivity between the hearing and hearing-impaired community. It also allows people new to sign language to ensure their signs are correct as well as check their signs with the text to sign language feature.")

def Text_to_sign():
    st.title("Text to Sign Language Translator")
    st.divider()
    st.markdown("This is the translator page. Here you can translate text to sign language.")
    
    
    if st.session_state.clear_input:
        st.session_state.to_be_translated = ""
        st.session_state.translating = False
        st.session_state.clear_input = False
        st.rerun()


    to_be_translated = st.text_input(
        "Enter text to translate to sign language! :", 
        disabled = st.session_state.translating, 
        on_change = translate_callback, key="to_be_translated"
        )
    
    retranslate = False
    if st.session_state.translating:
        if st.button("Translate Again?"):
            st.session_state.clear_input = True
            st.rerun()
    
    if st.session_state.translating and not retranslate:
        with st.spinner():
            time.sleep(2)
            process1 = to_be_translated.split()
            for i in process1:
                if not i.isalpha():
                    st.error("Please enter valid text containing only alphabetic characters.")
                else:
                    signlist.clear()
                    for i in to_be_translated.split():
                        signlist.append(i.upper())
                    print(signlist)
                    st.success(f"Translation complete: {st.session_state.to_be_translated} to sign language!")
                    for i in signlist:
                        for j in i:
                            image_list.append(sign_directory + f"\{j}.png")
                        st.image(image_list , width=100)
                        image_list.clear()

                    break
            if retranslate:
                st.session_state.clear_input = True
                st.rerun()
    
    st.markdown("Have additional words that you want as a singular sign? Enter them here and we will try to add them!")
    want_to_add = st.text_input("Enter additional words here:", key="want_to_add")
    if st.button("Submit"):
        if want_to_add is not None and want_to_add.isalpha():
            with st.spinner("Submitting your request..."):
                with open ("additional_words.txt", "a") as file:
                    file.write(f"{want_to_add}\n")
                st.success(f"Your request for adding '{want_to_add}' has been submitted! Thank you for your contribution!")
        else:
            st.error("Please enter a word to submit.")

def Sign_to_text():
    st.title("Sign Language Recognition")
    st.divider()
    st.markdown("This is the sign language recognition page. Here you can recognize sign language from a video feed using Machine Learning.")
    st.title("Webcam Classifier with Inverted Image and Border")
    webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    )


def help():
    st.title("Help")
    st.divider()
    st.markdown("This is the help page. Here you can find help and support from the Gemini AI!")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ["ai", "Hello! I am your Gemini AI assistant provided by Group I17. How may I assist you today?"]
        ]
    
    for role, message in st.session_state.chat_history:
        st.chat_message(role).markdown(message)
    
    user_input = st.chat_input(
        placeholder = "Enter your inquiry here...",
        key = "user_input",
    )
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append(["user", user_input])

        response = client.models.generate_content(
            model = "gemini-2.0-flash",
            config = genai.types.GenerateContentConfig(
                system_instruction = "" \
                "You are a helpful assistant. Please answer the user's questions to the best of your ability. You are in a chat interface of a website. The sidebar has a few features such as Text to Sign Language Translator, Sign Language to Text Translator, and Help. The user can ask you questions about these features. To use Text to sign, they need  to type the text and the code will display the pictures of the sign in sign alphabets. If they wish to resubmit just press the resubmit button."
            
            ),
            contents = st.session_state.chat_history
        )
        ml_response = response.text
        st.chat_message("ai").write(ml_response)
        st.session_state.chat_history.append(["ai", ml_response])


pages = {
    "Dashboard": [
        st.Page(home, title = "Home", icon="🏠"),
    ],
    "Translator": [
        st.Page(Text_to_sign, title = "Text to Sign Language", icon="🔄"),
        st.Page(Sign_to_text, title = "Sign Language to Text", icon="🔄"), 
    ],
    "Support": [
        st.Page(help, title = "Help", icon="❓"),
    ]   
}



pg = st.navigation(pag
