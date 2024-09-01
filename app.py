import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import google.generativeai as genai
import os
import dotenv

dotenv.load_dotenv() 
apiqdrant = os.getenv('QDRANT')
api_key_gmnini = os.getenv('GEMINI_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')

st.title('Sistema de Recuperação de Informações Jurídicas com RAG')

if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = TextEmbedding("snowflake/snowflake-arctic-embed-l")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=apiqdrant,
)
collection_name = "qdrantrag100"

query = st.text_input('Digite sua query:')
if query:
    st.write(f'Consultando Qdrant para: {query}')
    query_embedding = list(st.session_state.embedding_model.embed([query]))[0]

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=3,
    )

    st.write('Resultados da busca:')
    for result in search_results:
        st.write(f"**Score**: {result.score}")
        st.write(f"**Source**: {result.payload['source']}")
        st.write(f"**Text**: {result.payload['text']}\n")

    retrieved_docs = [doc.payload['text'] for doc in search_results]
    combined_text = "\n\n".join(retrieved_docs)

    # Iniciar o chat
    def start_chat():
        """Inicia um chat com o modelo Gemini, fornecendo um contexto inicial."""
        genai.configure(api_key=api_key_gmnini)
        model = genai.GenerativeModel('gemini-1.5-flash')
        conversation = model.start_chat()

        # Sys prompt
        conversation.send_message(f"""Você é um assistente especializado em Recuperação de Informação Aumentada (RAG), focado em fornecer respostas baseadas exclusivamente no conteúdo recuperado pelo modelo retriever. Seu objetivo é utilizar os trechos de texto mais relevantes, fornecidos a partir de uma consulta do usuário, como contexto para responder de forma precisa e objetiva. Os trechos recuperados estão relacionados a documentos do Conselho Administrativo de Defesa Econômica (CADE), uma entidade federal responsável por fiscalizar práticas econômicas no Brasil. Todas as suas respostas devem ser juridicamente fundamentadas, refletindo as informações contidas nos trechos fornecidos. Por favor, evite conjecturas e responda apenas com base nas informações disponíveis nos trechos recuperados. Aqui estão os trechos recuperados para sua referência: {combined_text}                     
        """)

        if 'conversation' not in st.session_state:
            st.session_state.conversation = conversation

    if st.button('Detalhamento dos termos retornados'):
        start_chat()

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