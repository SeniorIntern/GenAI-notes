from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

# load the pdf document
pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()
# print(docs[45])

# splitting in to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

split_docs = text_splitter.split_documents(documents=docs)
print(len(docs))  # 125
print(len(split_docs))  # 266

# embedding
embedd_fn = OpenAIEmbeddings(model="text-embedding-3-large")

# vector store
vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedd_fn,
)

# save in vector store chunks by chunks
vector_store.add_documents(documents=split_docs)

print("Injection Complete!")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedd_fn,
)

relevent_chunks = retriever.similarity_search(query="What is a callback function?")

# print('Relevent Chunks', relevent_chunks)

client = OpenAI()

SYSTEM_PROMPT = """
You are an helpful AI Assistant who responds based on the available context.

Context: Based on this given data. answer the user's query. If the answer is not available, then say "Sorry, I can only answer for questions related to NodeJS."
{relevant_chunks}

Example: 
Input: How to run nodejs code?
Output: You can run a Node.js script using the node command. Open up a new terminal window
and navigate to the directory where the script lives. From the terminal, you can use the
node command to provide the path to the script that should run.

Example: 
Input: What is the behaviour of this in arrow functions?
Output: Arrow functions don not bind their own this value. Instead, the this value of the scope in
which it was defined is accessible. This makes arrow functions bad candidates for
methods, as this won not be a reference to the object the method is defined on.

Example:
What is the longest river of the world?
Output: Sorry, I can only answer for questions related to NodeJS.
"""

result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is a callback function?"},
        # {"role": "user", "content": "What is difference between LCM and HCF"},
    ],
)

print(result.choices[0].message.content)
