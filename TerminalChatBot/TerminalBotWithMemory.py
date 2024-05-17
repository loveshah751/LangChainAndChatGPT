from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

chatMemory = ConversationBufferMemory(memory_key="messages", return_messages=True)

ChatPrompt = ChatPromptTemplate(
    input_variables=["Content"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ])

chain = LLMChain(
    llm=model,
    memory=chatMemory,
    prompt=ChatPrompt
)

while True:
    content = input("<<<< ")
    result = chain({"content": content})
    print("<<< AI Model: ", result["text"])
