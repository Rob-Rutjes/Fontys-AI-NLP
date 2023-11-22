import streamlit as st

from tempfile import NamedTemporaryFile
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool

# Initialize the tools to be used by the agent from the tools.py file
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversation_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",  # Key to be used to store the memory in the agent's memory
    k=5,                        # Number of messages to be stored in the memory
    return_messages=True        # If true, will return the messages stored in the memory when the agent is initialized
)

llm = ChatOpenAI(
    openai_api_key='sk-XlXfFVURP9Ka......',  # OpenAI API key, you can get one from https://platform.openai.com/
    temperature=0,                           # Temperature for the generation (how much the model should be creative)
    model_name='gpt-3.5-turbo'               # OpenAI Model name
)

agent = initialize_agent(
    agent="chat-conversational-react-description",  # Agent name
    tools=tools,                                    # Tools to be used by the agent
    llm=llm,                                        # Language model to be used by the agent
    max_iterations=5,                               # Maximum number of iterations
    verbose=True,                                   # Verbose mode, if true, will print the agent's messages to the console
    memory=conversation_memory,                     # History memory to be used by the agent
    early_stopping_method="generate"                # Early stopping method, if generate, will stop when the agent generates a message
)

# Streamlit app title and header
st.title('As a question about an image')
st.header('Upload your image below:')

# Upload button for the image
uploaded_file = st.file_uploader("", type=["jpeg", "jpg", "png"])

# If an image is uploaded
if uploaded_file:
    # Display the image
    st.image(uploaded_file, use_column_width=True)

    # Text input for the question
    question = st.text_input('Ask a question about the image')

    # Save file to a temporary file
    with NamedTemporaryFile(dir='.') as f:
        f.write(uploaded_file.getbuffer())
        image_path = f.name

        # If a question is asked
        if question and question != '':
            # Send the question and image path to the agent and await a response
            response = agent.run('{}, this is the image path: {}').format(question, image_path)

            # Display a spinner while the agent is working
            with st.spinner(text="IN PROGRESS..."):
                # Display the response
                st.write(response)
