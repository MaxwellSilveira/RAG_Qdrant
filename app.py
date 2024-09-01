import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct
from fastembed import TextEmbedding
import google.generativeai as genai
import os
import dotenv
from tqdm import tqdm

# Configurar chaves de API
apiqdrant = os.getenv('QDRANT')
api_key_gmnini = os.getenv('GEMINI_API_KEY')

# Inicializar Streamlit
st.title('Sistema de Recuperação de Informações Jurídicas com RAG')

# Carregar o arquivo CSV com os chunks
csv_path = 'C:/Users/Maxwell/Downloads/projeto_streamlit_cade/df100.csv'
st.write('Carregando arquivo CSV...')
df = pd.read_csv(csv_path)

# Criar lista de documentos
st.write('Criando lista de documentos...')
products_listdf = [Document(page_content=row['Conteudo_Chunk'], metadata={"source": row['Nome_Arquivo']}) for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processando documentos")]

# # Limitar a quantidade de documentos para teste
# products_listdf = products_listdf[:100]

# Mostrar um exemplo de documento
if st.checkbox('Exibir exemplo de documento'):
    st.write(products_listdf[0].page_content)

# Inicializar o modelo de embeddings Snowflake
st.write('Gerando embeddings...')
embedding_model = TextEmbedding("snowflake/snowflake-arctic-embed-l")
texts = [doc.page_content for doc in products_listdf]
embeddings = embedding_model.embed(texts)

# Conectar ao Qdrant
qdrant_client = QdrantClient(
    url="356af5af-fd28-4b0b-a894-09c358e6b1e0.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=apiqdrant,
)

collection_name = "qdrantrag100"

# Inserir os documentos no Qdrant
st.write('Inserindo documentos no Qdrant...')
points = [
    PointStruct(
        id=idx,
        vector=embedding,
        payload={"text": doc.page_content, "source": doc.metadata["source"]},
    )
    for idx, (embedding, doc) in enumerate(zip(embeddings, products_listdf))
]
qdrant_client.upsert(collection_name=collection_name, points=points)

# Fazer uma query
query = st.text_input('Digite sua query:')
if query:
    st.write(f'Consultando Qdrant para: {query}')
    query_embedding = embedding_model.embed([query])[0]

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

    # Combinar textos dos resultados
    retrieved_docs = [doc.payload['text'] for doc in search_results]
    combined_text = "\n\n".join(retrieved_docs)

    # Inicializar a conversa com o modelo Gemini
    if st.button('Iniciar Conversa com Gemini'):
        genai.configure(api_key=api_key_gmnini)
        model = genai.GenerativeModel('gemini-1.5-flash')
        conversation = model.start_chat()
        response = conversation.send_message(f"Você é um assistente do Augmented Generation Recovery (RAG) e fará uma conversa relacionada ao que foi retornado pelo modelo do retriever. Nesse caso, você receberá os chunks com a maior similaridade com a consulta do usuário e os usará como contexto para formular respostas sobre os chunks retornados. Você receberá perguntas sobre os seguintes chunks retornados de um retriever RAG: {combined_text}. Responda as perguntas de forma clara e concisa, utilizando as informações presentes no texto. Sabendo que são documentos do CADE (Conselho Administrativo de Defesa Econômica) um orgão federal responsável por fiscalizar, então traga as respostas ao usuário trazendo um viés jurídico apenas com as informações presentes nos chunks")

        st.write("Gemini:", response.text)

        user_input = st.text_input("Você:", "")
        if user_input:
            response = conversation.send_message(user_input)
            st.write("Gemini:", response.text)
