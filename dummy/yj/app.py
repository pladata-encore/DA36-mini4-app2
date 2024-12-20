import streamlit as st
from streamlit_chat import message as msg
import openai
import speech_recognition as sr
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
import os

# .env 파일에서 API 키 읽기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 페이지 설정
st.set_page_config(
    page_title="🐎 마! 챗봇",
    page_icon="🎨",
    layout="wide",
)

# 배경 이미지 업로드 및 설정
background_image = st.file_uploader("배경 이미지 업로드", type=["png", "jpg", "jpeg"])

def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

if background_image is not None:
    img = Image.open(background_image)
    bg_base64 = get_image_base64(img)
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url('data:image/png;base64,{bg_base64}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stButton > button {{
            background-color: #007BFF;
            color: white;
            border-radius: 5px;
            padding: 10px;
        }}
        .stTextInput > div > input {{
            border: 2px solid #007BFF;
            border-radius: 5px;
        }}
        .chat-container {{
            border: 2px solid #007BFF;
            border-radius: 10px;
            padding: 20px;
            background-color: rgba(255, 255, 255, 0.9); /* 칠판처럼 보이게 */
            max-height: 500px;
            overflow-y: scroll;
            margin-bottom: 20px;
        }}
        .chat-board {{
            border: 5px solid #222222;
            border-radius: 15px;
            padding: 10px;
            background-color: #222222;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# 제목과 설명
st.title("🐎 마! 챗봇!")
st.write("친절하고 유용한 대답을 제공하는 챗봇입니다. 무엇이든 물어보세요!")

# 대화 섹션
st.subheader("대화하기")
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "안녕하세요! 무엇을 도와드릴까요?"}
        ]

    # 칠판처럼 구분된 대화창
    st.markdown('<div class="chat-board">', unsafe_allow_html=True)
    for i, message in enumerate(st.session_state["messages"]):
        if message["role"] == "user":
            msg(message["content"], is_user=True, key=f"user_{i}")
        else:
            msg(message["content"], is_user=False, key=f"assistant_{i}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# 음성 입력 기능
recognizer = sr.Recognizer()
microphone = sr.Microphone()

if st.button("음성 입력 시작"):
    with microphone as source:
        st.write("음성을 기다리는 중...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio, language="ko-KR")
            st.write(f"음성으로 입력된 내용: {user_input}")
            st.session_state["messages"].append({"role": "user", "content": user_input})
        except sr.UnknownValueError:
            st.write("음성을 이해할 수 없습니다. 다시 시도해 주세요.")
        except sr.RequestError:
            st.write("음성 인식 서비스에 문제가 발생했습니다. 나중에 다시 시도해 주세요.")

# 텍스트 입력
user_input = st.text_input("질문을 입력하세요", "")
if st.button("질문하기"):
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # OpenAI API 호출
    if user_input:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=user_input,
            max_tokens=150
        )
        assistant_response = response.choices[0].text.strip()
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

st.markdown("---")
st.info("Streamlit 앱을 실행하려면, `streamlit run app.py` 명령을 실행하세요.")
