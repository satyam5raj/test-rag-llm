import os
from dotenv import load_dotenv

load_dotenv()

from rag.embedder import get_query_embedding
from rag.vectordb import get_relevant_docs
from rag.prompt_builder import build_prompt
from rag.llm import ask_llm
from data_importer import load_real_data

# Load real data (this function handles adding to ChromaDB internally)
print("🚀 Initializing RAG System with Real Data...")
load_real_data()

print("\n⚽ RAG Football Assistant Ready!")
print("Ask me anything about football!")
print("Type 'quit', 'exit', or 'q' to stop")
print("-" * 60)

while True:
    try:
        # Get user input
        query = input("\n🤔 Your question: ").strip()
        
        # Check if user wants to quit
        if query.lower() in ['quit', 'exit', 'q']:
            print("⚽ Thanks for using the RAG system! Goodbye! 👋")
            break
        
        # Skip empty queries
        if not query:
            print("Please enter a question.")
            continue
        
        print("🔍 Processing your question...")
        
        # Process the query
        query_embedding = get_query_embedding(query)
        docs = get_relevant_docs(query_embedding, k=5)  # Get more relevant docs
        prompt = build_prompt(docs, query)
        answer = ask_llm(prompt)
        
        # Display results
        print(f"\n❓ Q: {query}")
        print(f"🤖 A: {answer}")
        print("⚽" + "-" * 59)
        
    except KeyboardInterrupt:
        print("\n\n⚽ Goodbye! Thanks for using the Football Assistant! 👋")
        break
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please try again.")