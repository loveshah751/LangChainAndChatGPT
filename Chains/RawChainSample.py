# Chain is consist of Prompt + LLM Model

# 1 Import Prompt Template from langchain

# 2 Import LLM chain from langchains

# 3 create Prompt Template

# 4 create LLMChain with prompt and llm model

# 5 run the LLMChain with dictionary of inputs variables

# 6 use python dotEnv package to secure the API Key

# 7 use argParse to parse the commandLine arguments

# 8 changing the result output value from Text to custom String

# 9 in LLMChain class accept the prompt and llm model. Also this class has the properies like input variables and output variables.

# 10 creating another Chain for writing the Tests for output from previous code generation

# 11 to connect these two chains we need squentialChain from the langChain.chains

# 12 in SequentialChain, we will define our chains in the order of executes and input& output variables
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains import SequentialChain
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="java")
args = parser.parse_args()

load_dotenv()

llm = OpenAI()
codeGenerationPrompt = PromptTemplate(
    template="Write a very short {language} function that will {task}.",
    input_variables=["language", "task"]
)

testGenerationPrompt = PromptTemplate(
    template="write a test case in {language} to test the code:{code}",
    input_variables=["language", "code"]
)

codeGenerationChain = LLMChain(
    llm=llm,
    prompt=codeGenerationPrompt,
    output_key="code"
)

testGenerationChain = LLMChain(
    llm=llm,
    prompt=testGenerationPrompt,
    output_key="test"
)

sequentialChains = SequentialChain(
    chains=[codeGenerationChain, testGenerationChain],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)

result = sequentialChains({
    "language": args.language,
    "task": args.task
})

# result = codeGenerationChain({
#     "language": args.language,
#     "task": args.task
# })

# print("User Prompt:", args.task, "\n",
#       "Programming Language used:", result["language"],"\n",
#       "Code Generation by Open AI LLM:\n", result["code"], "\n",
#       "Test case Generation by Open AI LLM:\n", result["test"])


print(">>>>>>>> User Prompt:")
print(args.task)


print(">>>>>>>> GENERATED CODE BY LLM:")
print(result["code"])


print(">>>>>>>> GENERATED Test BY LLM:")
print(result["test"])



