from fastembed import TextEmbedding
import google.generativeai as genai
import dotenv
import streamlit as st

dotenv.load_dotenv()

class ModelManager:
    def __init__(self):
        self.embedding_model = None
        self.conversation = None

    def load_embedding_model(self):
        if 'embedding_model' not in st.session_state:
            st.session_state.embedding_model = TextEmbedding("snowflake/snowflake-arctic-embed-l")
        return st.session_state.embedding_model

    def load_conversation(self, api_key):
        if 'conversation' not in st.session_state:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            conversation = model.start_chat()
            st.session_state.conversation = conversation
        return st.session_state.conversation

model_manager = ModelManager()
