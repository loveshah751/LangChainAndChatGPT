# Chain in Langchain is the squence of call to either LLM model, tool or PreProcessing Step.

# 1 Import Prompt Template from langchain
from langchain.prompts import PromptTemplate

# 2 Import LLM chain from langchains
from langchain.chains import LLMChain

# 3 use python dotEnv package to secure the API Key
from dotenv import load_dotenv

# 4 import LLM model like OpenAI, Bard etc...
from langchain.llms import OpenAI

# 5 read the .env file for API key
load_dotenv()

# 6 create Prompt Template
myPrompt = PromptTemplate(
    template="Write a very short {language} function that will {language}.",
    input_variables=["language", "task"]
)
# 7 create a llm model instance
myllmInstance = OpenAI()

# 8 create LLMChain with prompt and llm model
myLLMChain = LLMChain(
    llm=myllmInstance,
    prompt=myPrompt
)

# 9 run the LLMChain with dictionary of inputs variables
result = myLLMChain({
    "language": "python",
    "task": "return a list of numbers"
})

# 10 print the result, which will have Result[Input Variables, text(Result)]
print("\n", result)
# 11 print the result from the LLM model
print("<<<<<<<<<<< LLM Model Result")
print(result["text"])
