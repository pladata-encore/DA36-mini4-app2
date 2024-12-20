import streamlit as st
from streamlit_chat import message as msg
import open_api


def main():
    st.set_page_config(
            page_title='마! 쳇봇',
            page_icon="🐎",
            layout='wide'
        )
    st.header('🐎horse! chatbot🐎')
    st.markdown('---')

    with st.expander('🐎horse! chatbot🐎 프로그램을 사용하는 방법', expanded=False):
        st.write(
            """
            1. 질문하기 버튼을 눌러 질문을 입력하세요!
            2. 질문을 하면 내용에 맞게 답변이 산출됩니다.
            3. LLM은 OpenAI사의 GPT모델을 사용합니다.
            """
    )
    system_instruction = '당신은 친절한 챗봇입니다.'

    # session state 초기화
    # session_state은 streamlit에서 제공되는 저장을 위한 일종의 빈 dict 형태
    # - messages: LLM 질의/웹페이지 시각화를 위한 대화내역
    # - check_reset: 초기화를 위한 flag
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {'role': 'system', 'content': system_instruction}
        ]
    if 'check_reset' not in st.session_state:
        st.session_state['check_reset'] = False
    with st.sidebar:
        model = st.radio(label='GPT 모델', options=['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o'], index=2)
        print(model)
        if st.button(label='초기화'):
            st.session_state['messages'] = [
                {'role': 'system', 'content': system_instruction}
            ]
            st.session_state['check_reset'] = True  # 화면 정리
    st.markdown('---')
    st.subheader('질문/답변')
    #if (audio.duration_seconds > 0) and (st.session_state['check_reset'] == False):
    for i, message in enumerate(st.session_state['messages']):
        role = message['role']
        content = message['content']
        if role == 'user':
            msg(content, is_user=True, key=str(i))
        elif role == 'assistant':
            msg(content, is_user=False, key=str(i))

        else:
            # 초기화버튼 누르면, 화면이 정리되고, 다시 check_reset을 원상복구
            st.session_state['check_reset'] = False


if __name__ == '__main__':
    main()