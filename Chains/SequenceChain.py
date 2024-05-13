from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import argparse

load_dotenv()
llmModel = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return sum of two numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# create the code generation chain
codeGenerationPrompt = PromptTemplate(
    template="write  a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

codeGenerationChain = LLMChain(
    llm=llmModel,
    prompt=codeGenerationPrompt,
    output_key="code"  # way to override the default value text.
)

# creating the test generation chain
testCaseGenerationPrompt = PromptTemplate(
    template="write a test case in {language} to test the code:{code}",
    input_variables=["language", "code"]
)

testCaseGenerationChain = LLMChain(
    llm=llmModel,
    prompt=testCaseGenerationPrompt,
    output_key="test"
)

# Connecting the above two chains together
sequentialChains = SequentialChain(
    chains=[codeGenerationChain, testCaseGenerationChain],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)

# running the sequential Chain to get the results from LLM model
result = sequentialChains({
    "language": args.language,
    "task": args.task
})

# printing the result from the LLM model
print("<<<<< Code generation by LLM Model: \n")
print(result["code"])

print("<<<<< Test case generation by LLM Model: \n")
print(result["test"])
