
from ollama import chat, embed

# Test 1: Generate an answer with the local LLM
response = chat(
    model="qwen3:1.7b",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in two simple sentences."
        }
    ],
)

print("\n--- LLM TEST ---")
print(response.message.content)

# Test 2: Generate an embedding for text
embedding_response = embed(
    model="nomic-embed-text",
    input="RAG helps an AI answer questions using personal documents.",
)

print("\n--- EMBEDDING TEST ---")
print("Number of embedding values:", len(embedding_response.embeddings[0]))
print("Embedding generated successfully.")
