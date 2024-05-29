from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from chromadb import Settings
from Embeddings.CustomRetriever import RedundantFilterRetriever

# Run the ChromaDB Embeddings file, to create the embeddings before running this file.....
load_dotenv()
embeddings = OpenAIEmbeddings()

chromaDBSettingsForDocker = Settings(
    chroma_api_impl="chromadb.api.fastapi.FastAPI",
    chroma_server_host="localhost",
    chroma_server_http_port=8090

)

db = Chroma(
    embedding_function=embeddings,
    client_settings=chromaDBSettingsForDocker
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

results = retriever.get_relevant_documents("What is interesting fact about the English language?")

for result in results:
    print("\n")
    print(result.page_content)
