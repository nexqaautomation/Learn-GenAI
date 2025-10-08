from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

# 1. Define the chat model
model = ChatOpenAI(temperature=0)

# 2. Create a static prompt (no placeholders)
prompt = PromptTemplate.from_template(
    "Write a welcome email to a new customer."
)

# 3. Build the chain
chain = prompt | model

# 4. Run the chain (no inputs needed)
response = chain.invoke({}) 

print(response.content)