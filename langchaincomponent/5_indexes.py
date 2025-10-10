from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# Step 1a - Load PDF
loader = PyPDFLoader("ai.pdf")
docs = loader.load()

print(f"\n------------------------------------- Chunks length -------------------------------------")
print(len(docs))

# Step 1b - Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

for i, chunk in enumerate(chunks, start=1):
    print(f"\n------------------------------------- Chunk {i} -------------------------------------")
    print(chunk.page_content)

# Step 1c and 1d - Embedding Generation and Storing in Vector Store
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = FAISS.from_documents(chunks, embeddings)

print(f"\n------------------------------------- Vector Store -------------------------------------")
print(vector_store.index_to_docstore_id)

# Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

question = 'What is the difference between AI, ML, deep learning, and NLP?'
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

print(f"\n------------------------------------- Context -------------------------------------")
print(context_text)

# Step 3 - Augmentation
prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
""",
    input_variables=['context', 'question']
)

final_prompt = prompt.invoke({"context" : context_text, "question" : question})

print(f"\n------------------------------------- Final Prompt -------------------------------------")
print(final_prompt)

# Step 4 - Generation

llm = ChatOpenAI()
answer = llm.invoke(final_prompt)

print(f"\n------------------------------------- Output -------------------------------------")
print(answer.content)