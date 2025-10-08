from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. Create a reusable prompt template
prompt = PromptTemplate.from_template(
    "Write a polite email reply to: {customer_message}"
)

# 2. Define the model
model = ChatOpenAI()

# 3. Create a parser
parser = StrOutputParser()

# 4. Build the chain (Prompt → Model)
chain = prompt | model | parser

# 5. Invoke the chain
response = chain.invoke({"customer_message": "I want to restart my subscription."})

print(response)