from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool

tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversation_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key='sk-XlXfFVURP9Ka......',
    temperature=0,
    model_name='gpt-3.5-turbo'
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversation_memory,
    early_stopping_method="generate"
)

st.title('As a question about an image')

st.header('Upload your image below:')

uploaded_file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, use_column_width=True)

    question = st.text_input('Ask a question about the image')

    with NamedTemporaryFile(dir='.') as f:
        f.write(uploaded_file.getbuffer())
        image_path = f.name

        if question and question != '':
            response = agent.run('{}, this is the image path: {}').format(question, image_path)

            with st.spinner(text="IN PROGRESS..."):
                st.write(response)
