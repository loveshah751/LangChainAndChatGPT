from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()

res = llm("write very small punjabi Song")
print(res)
