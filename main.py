import streamlit as st
from models import model_manager
from qdrant_utils import search_qdrant
from chat_utils import start_chat
import os

st.title('Sistema de Recuperação de Informações Jurídicas com RAG')

# Carregar modelos
embedding_model = model_manager.load_embedding_model()
api_key_gmnini = os.getenv('GEMINI_API_KEY')

# Conectar ao Qdrant
collection_name = "qdrantrag100"

query = st.text_input('Digite sua query:')
if query:
    st.write(f'Consultando Qdrant para: {query}')
    search_results = search_qdrant(query, embedding_model, collection_name)

    st.write('Resultados da busca:')
    for result in search_results:
        st.write(f"**Score**: {result.score}")
        st.write(f"**Source**: {result.payload['source']}")
        st.write(f"**Text**: {result.payload['text']}\n")

    retrieved_docs = [doc.payload['text'] for doc in search_results]
    combined_text = "\n\n".join(retrieved_docs)

    if st.button('Detalhamento dos termos retornados'):
        conversation = model_manager.load_conversation(api_key_gmnini)
        start_chat(conversation, combined_text)

    if 'conversation' in st.session_state:
        user_input = st.text_input("Digite sua pergunta:")
        st.caption("Digite 'sair' para encerrar a sessão.")
        if user_input:
            if user_input.lower() == "sair":
                st.write("Conversa encerrada.")
                del st.session_state.conversation 
            else:
                response = st.session_state.conversation.send_message(user_input)
                st.write("Gemini:", response.text)
