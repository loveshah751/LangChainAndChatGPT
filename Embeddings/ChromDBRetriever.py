from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from chromadb import Settings

load_dotenv()
embeddings = OpenAIEmbeddings()

chromaDBSettingsForDocker = Settings(
    chroma_server_host="localhost",
    chroma_server_http_port=8090,
    chroma_api_impl="chromadb.api.fastapi.FastAPI"
)

chromaDB = Chroma(
    client_settings=chromaDBSettingsForDocker,
    embedding_function=embeddings
)

# Here we are using ChromDB retriever for retrieving the embeddings, rather then creating and getting.
chromaDBRetriever = chromaDB.as_retriever()

results = chromaDBRetriever.get_relevant_documents("What is interesting fact about the English language?")

for result in results:
    print("\n")
    print(result.page_content)
