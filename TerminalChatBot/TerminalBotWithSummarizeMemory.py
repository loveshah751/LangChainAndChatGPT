from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# we are using this specific type of Memory because we don't to store the whole chat history in the memory,
# it will overload memory at some point of Time. instead we will use ConversationSummaryBufferMemory class which
# create the summary of our previous Conservations and store it for future reference. ConversationSummaryBufferMemory
# has it own internal prompt, chain and uses LLM model to summarize the the previous chat.
# uncomment the verbose property to view the internal chain execution by ConversationSummaryBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(verbose=True)

chatMemory = ConversationSummaryBufferMemory(memory_key="messages", return_messages=True, llm=model)

ChatPrompt = ChatPromptTemplate(
    input_variables=["Content"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ])

chain = LLMChain(
    llm=model,
    memory=chatMemory,
    prompt=ChatPrompt,
    verbose=True
)

while True:
    content = input("<<<< ")
    result = chain({"content": content})
    print("<<< AI Model: ", result["text"])
