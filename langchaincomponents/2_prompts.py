from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Creates a reusable prompt template with a placeholder.
prompt_template = PromptTemplate.from_template(
    "Write a polite email reply to: {customer_message}"
)

# Invoke the prompt
formatted_prompt = prompt_template.invoke({"customer_message": "I want to restart my subscription."})
print(formatted_prompt)

#Create a model
model = ChatOpenAI()

#Invoke the model
response = model.invoke(formatted_prompt)

print(response.content)