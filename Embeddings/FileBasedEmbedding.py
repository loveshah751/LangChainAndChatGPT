from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

textLoader = TextLoader("../facts.txt")

loadedText = textLoader.load()
print("Simple loading the file using LangChain loader... \n ")
print(loadedText)
print("\n")

textSplitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

print("Loading and Splitting the text using load_and_split func from LangChain TextLoader.\n")

embeddingCreator = OpenAIEmbeddings()

docs = textLoader.load_and_split(text_splitter=textSplitter)

first_document = docs[0].page_content

embeddingForFirstDocument = embeddingCreator.embed_query(first_document)

print("Embeddings for first document from the fact file\n")

print(
    "*********************************************************************************************************************************************************************")

print(embeddingForFirstDocument)

print(
    "*********************************************************************************************************************************************************************")

print("\n", "printing with For Loop")

for doc in docs:
    # commenting this line, to avoid the embedding call to OPENAI, and it cost some dollar value
    # it is not worth it until we have our dataStore, to store this embedding rather then call the OPENAI everyTime, on program run
    #embeddingCreator.embed_query(doc.page_content)
    print(doc.page_content)
    print("\n")
