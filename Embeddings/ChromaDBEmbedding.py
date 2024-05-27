from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from chromadb.config import Settings

load_dotenv()

embeddings = OpenAIEmbeddings()

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)
loader = TextLoader("../facts.txt")

# Spin up the docker image for chromeDB
chromaDbDockerConnection = Settings(
    chroma_server_host="localhost",
    chroma_server_http_port="8090",
    # persist_directory="emb" // for local storage within the directory
    chroma_api_impl="chromadb.api.fastapi.FastAPI",  # Storing in docker container
)

docs = loader.load_and_split(text_splitter=splitter)
db = Chroma.from_documents(
    embedding=embeddings,
    documents=docs,
    client_settings=chromaDbDockerConnection
)

results = db.similarity_search("What is interesting fact about the English language?")

for result in results:
    print("\n")
    print(result.page_content)
