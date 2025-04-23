import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# vocab size -  total unique tokens in a given model
gpt4o_encoder = tiktoken.encoding_for_model("gpt-4o")
print("gpt 4o vocab size=", gpt4o_encoder.n_vocab)  # 200019

gpt4_encoder = tiktoken.encoding_for_model("gpt-4")
print(f"gpt 4 vocab size= {gpt4_encoder.n_vocab}")  # 100277

# vector embedding - semantic meaning of word in numeric form
# text embedding
client = OpenAI()

text = "Eiffel Tower is in Paris and is a famous landmark, it is 324 meters tall"

# Generate text's vector embeddings
response = client.embeddings.create(input=text, model="text-embedding-3-small")

print("Vector embeddings", response.data[0].embedding)
