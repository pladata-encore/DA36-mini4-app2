import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# .env 파일 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI API 키 환경 변수 설정
os.environ["OPENAI_API_KEY"] = openai_api_key


# OpenAI 모델 초기화
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

vectorstore_paths = {
    "경마정보": r"C:\Workspace\DA36_mini4_ma\min\vectors_new\vs_race_guide",
    "경주일정": r"C:\Workspace\DA36_mini4_ma\min\vectors_new\vs_schedule",
    "우승마기록": r"C:\Workspace\DA36_mini4_ma\min\vectors_new\vs_winners",
    "경주마정보": r"C:\Workspace\DA36_mini4_ma\min\vectors_new\vs_horse_info"
}


def summarize_query(query):
    if len(query.split()) <= 20:
        return False, query

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are an assistant that summarizes questions to korean into concise queries for search."),
        HumanMessage(content=f"Original question: {query}\n\nSummarize this into a concise query:")
    ])
    summarized_query = (prompt | model | StrOutputParser()).invoke({"query": query})
    return True, summarized_query


def classify_and_summarize_query(query):
    is_summarized, processed_query = summarize_query(query)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""\
        You are an expert in horse racing. Your task is to classify the given user question into one of the following categories:

        - 경마정보: Questions about general information such as rules, betting methods, and terminology.
        - 경주일정: Questions about race schedules, dates, times, or locations. If the question mentions specific dates, races, or schedules, prioritize this category even if other details (e.g., horse performance) are included.
        - 우승마기록: Questions about winning horses and their records.
        - 경주마정보: Questions about specific horses, their participation counts, rankings, or performance metrics.

        If the question does not match any category, return "Unknown".
        """),
        HumanMessage(content=f"User Question: {processed_query}\n\nClassify this question into one of the categories:")
    ])

    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    category = chain.invoke({"query": processed_query}).strip()

    return category


def rag_and_prompt(query):
    category = classify_and_summarize_query(query)

    vectorstore_path = vectorstore_paths[category]
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    is_summarized, summarized_query = summarize_query(query)  # 요약 수행
    if is_summarized:
        print("⚠️질문이 20단어를 초과하여 요약되었습니다.")
        print(f"- 기존 질문: {query}\n- 요약된 질문: {summarized_query}\n")
    else:
        print(f"질문: {query}\n")

    results = retriever.get_relevant_documents(summarized_query)

    retrieved_data = "\n".join([doc.page_content for doc in results])

    # **대화 히스토리 전달**
    conversation_history = "\n".join(
        [f"👤 사용자: {msg['content']}" if msg["role"] == "user" else f"🤖 챗봇: {msg['content']}"
         for msg in st.session_state.messages]
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""\
            당신은 전문적이고 애교가 많은 경마 안내 챗봇입니다. 
            대화 기록을 참고하여 사용자의 질문에 대해 상세하고 재미있는 답변을 제공합니다.
            예시: 2024년 12월 21일 서울 경주 일정을 물어보면, 경주의 시간과, 최근 성적이 좋은 말의 정보 등을 알려줘야 합니다.
            """),
        HumanMessage(content=f"""\ 
            대화 기록:
            {conversation_history}
            사용자 질문: {query}
            context: {retrieved_data}
            사용자의 질문에 대해 검색된 정보와 대화 기록을 바탕으로 답변하세요.
            """)
    ])

    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    response = chain.invoke({"query": query})
    return response


def ask_question(query):
    response = rag_and_prompt(query)
    return response


# Streamlit 앱 시작

st.set_page_config(
    page_title="MA!",
    page_icon="🐎",
    # layout="wide",
)

main_bg_color = "#CEF6CE"
st.markdown(f"""
        <style>
        /* 메인 페이지 배경 색 설정 */
        .stApp {{
            background-color: {main_bg_color};
        }}
        </style>
        """, unsafe_allow_html=True)


st.title("🏇 경마 안내 챗봇 MA!")
st.write("환영합니다! 경마에 관한 질문을 남겨보세요. 실시간으로 답변해드릴게요! 😊")
st.divider()

# **대화 기록 초기화**
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 기록 초기화

# **질문 입력 필드 초기화**
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

st.write("🔖 대화 기록:")
pairs = []
for i in range(0, len(st.session_state.messages), 2):
    user_msg = st.session_state.messages[i]
    assistant_msg = st.session_state.messages[i + 1] if i + 1 < len(st.session_state.messages) else None
    pairs.append((user_msg, assistant_msg))

for user_msg, assistant_msg in pairs:
    st.write(f"🐼 민하: {user_msg['content']}")
    if assistant_msg:
        st.write(f"-`,馬ˎ´-❕: {assistant_msg['content']}")
    st.divider()  # 각 질문-답변 쌍 뒤에 구분선 추가

user_input = st.text_input("질문을 입력하세요:", value=st.session_state.current_question)

if st.button("궁금해요"):
    if user_input.strip():
        # 현재 질문 저장
        st.session_state.current_question = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 질문 요약 수행
        is_summarized, summarized_query = classify_and_summarize_query(user_input)
        if is_summarized:
            st.write("⚠️ 질문이 길어 요약되었습니다.")
        else:
            summarized_query = user_input

        # RAG 및 답변 생성
        with st.spinner("답변을 생성 중입니다..."):
            answer = rag_and_prompt(summarized_query)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        # UI에 요약된 질문과 답변 바로 출력
        st.write(f"🐼 민하: {user_input}")
        if is_summarized:
            st.write(f"🔍 요약된 질문: {summarized_query}")
        st.write(f"-`,馬ˎ´-❕: {answer}")
        st.divider()

        # 입력 필드 초기화
        st.session_state.current_question = ""
