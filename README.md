# Sistema de Recuperação de Informações Jurídicas com RAG

Este projeto é um sistema de recuperação de informações jurídicas baseado em Recuperação Aumentada por Geração (RAG). O objetivo é permitir que os usuários façam consultas sobre documentos do Conselho Administrativo de Defesa Econômica (CADE). O sistema utiliza um banco de dados Qdrant para armazenar e recuperar embeddings dos documentos e o modelo Gemini para gerar respostas baseadas em consultas do usuário.

## Descrição do Projeto

### Funcionalidade

- **Consulta de Documentos**: O usuário pode inserir uma consulta, e o sistema recupera os documentos mais relevantes armazenados no banco de dados Qdrant.
- **Geração de Respostas**: Utilizando o modelo Gemini, o usuário é capaz de realizar perguntas sobre os documentos retornados pelo modelo retriever.
- **Interação com o Usuário**: A interface é construída com Streamlit, permitindo interações simples e intuitivas.

### Estrutura dos Dados

Os dados consistem em 100 documentos do CADE, que foram divididos em chunks de 1024 tokens com um overlap de 128 tokens. Foi feito o embedding de cada chunk com o modelo Snowflakes-l que possui um bom desempenho para a língua portuguesa, o modelo possui suporte para criação de embeddings com até 1024 dimensões.

## Requisitos

- Python 3.12.3
- Streamlit
- Qdrant Client
- FastEmbed
- Google Generative AI
- dotenv

## Instalação

1. Clone o repositório:

    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DO_REPOSITORIO>
    ```

2. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

3. Configure suas variáveis de ambiente. Crie um arquivo `.env` na raiz do projeto e adicione as seguintes variáveis:

    ```env
    QDRANT=<URL_DO_QDRANT>
    QDRANT_URL=<URL_DO_QDRANT>
    GEMINI_API_KEY=<SUA_CHAVE_API_GEMINI>
    ```

## Estrutura do Código

### `main.py`

Este é o arquivo principal que utiliza Streamlit para criar a interface do usuário. Ele carrega os modelos, realiza consultas no Qdrant e gerencia a interação com o modelo Gemini.

### `models.py`

Contém a lógica para o carregamento e gerenciamento dos modelos de embeddings e chat. O `ModelManager` é responsável por garantir que os modelos sejam carregados apenas uma vez e reutilizados durante a sessão.

### `qdrant_utils.py`

Inclui funções para interagir com o banco de dados Qdrant, incluindo a busca de documentos baseados em embeddings.

### `chat_utils.py`

Contém funções para interagir com o modelo de chat Gemini, incluindo a inicialização da conversa e o envio de mensagens.

## Como Usar

1. Execute o aplicativo Streamlit:

    ```bash
    streamlit run main.py
    ```

2. Na interface do usuário:
   - **Digite sua consulta** no campo de texto para procurar documentos relevantes.
   - **Clique em "Detalhamento dos termos retornados"** para iniciar um chat com o modelo Gemini, que usa os documentos recuperados como contexto.
   - **Digite suas perguntas** no campo de entrada de chat para obter respostas baseadas no conteúdo dos documentos.

3. Para encerrar a sessão de chat, digite `sair`.

## Observações

- Certifique-se de que as credenciais da API estão corretamente configuradas no arquivo `.env`.
- O banco de dados Qdrant deve estar configurado e acessível conforme as configurações fornecidas.


## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).
