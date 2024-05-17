from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

chatModel = ChatOpenAI(verbose=True)

chatPrompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chatChain = LLMChain(
    llm=chatModel,
    prompt=chatPrompt,
    verbose=True
)

while True:
    content = input("<<< ")
    result = chatChain({"content": content})
    print("AI: ", result["text"])
