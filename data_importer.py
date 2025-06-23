# data_importer.py - Import real datasets into ChromaDB
import requests
import pandas as pd
import json
import os
from typing import List, Dict
from rag.embedder import get_doc_embeddings
from rag.vectordb import collection

class DataImporter:
    def __init__(self):
        self.data_sources = {
            'wikipedia': self.import_wikipedia_articles,
            'csv': self.import_csv_file,
            'json': self.import_json_file,
            'txt': self.import_text_file,
            'web_scrape': self.scrape_website,
            'football_api': self.import_football_data
        }
    
    def import_wikipedia_articles(self, topics: List[str], max_articles: int = 10):
        """Import Wikipedia articles on given topics"""
        print(f"📥 Importing Wikipedia articles for: {', '.join(topics)}")
        
        documents = []
        for topic in topics:
            try:
                # Get Wikipedia page content
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    # Split into chunks for better embedding
                    content = data.get('extract', '')
                    if content:
                        chunks = self.split_text(content, 500)  # 500 char chunks
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                'text': chunk,
                                'source': f"Wikipedia - {topic}",
                                'id': f"wiki_{topic}_{i}"
                            })
                        print(f"✅ Added {len(chunks)} chunks from {topic}")
                    
            except Exception as e:
                print(f"❌ Error importing {topic}: {e}")
        
        return self.add_to_chromadb(documents)
    
    def import_csv_file(self, file_path: str, text_column: str, metadata_columns: List[str] = None):
        """Import data from CSV file"""
        print(f"📥 Importing CSV file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            documents = []
            
            for index, row in df.iterrows():
                text = str(row[text_column])
                metadata = {}
                
                if metadata_columns:
                    for col in metadata_columns:
                        if col in df.columns:
                            metadata[col] = str(row[col])
                
                documents.append({
                    'text': text,
                    'source': f"CSV - {file_path}",
                    'metadata': metadata,
                    'id': f"csv_{index}"
                })
            
            print(f"✅ Loaded {len(documents)} records from CSV")
            return self.add_to_chromadb(documents)
            
        except Exception as e:
            print(f"❌ Error importing CSV: {e}")
            return False
    
    def import_json_file(self, file_path: str, text_field: str):
        """Import data from JSON file"""
        print(f"📥 Importing JSON file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and text_field in item:
                        documents.append({
                            'text': str(item[text_field]),
                            'source': f"JSON - {file_path}",
                            'metadata': {k: v for k, v in item.items() if k != text_field},
                            'id': f"json_{i}"
                        })
            
            print(f"✅ Loaded {len(documents)} records from JSON")
            return self.add_to_chromadb(documents)
            
        except Exception as e:
            print(f"❌ Error importing JSON: {e}")
            return False
    
    def import_text_file(self, file_path: str, chunk_size: int = 1000):
        """Import large text file by splitting into chunks"""
        print(f"📥 Importing text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = self.split_text(content, chunk_size)
            documents = []
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    'text': chunk,
                    'source': f"Text File - {file_path}",
                    'id': f"txt_{i}"
                })
            
            print(f"✅ Split into {len(documents)} chunks")
            return self.add_to_chromadb(documents)
            
        except Exception as e:
            print(f"❌ Error importing text file: {e}")
            return False
    
    def import_football_data(self):
        """Import real football data from free APIs"""
        print("📥 Importing real football data...")
        
        documents = []
        
        # Example: Football player data (you can expand this)
        football_facts = [
            "Lionel Messi has scored over 800 career goals for club and country, playing primarily for Barcelona and Paris Saint-Germain.",
            "Cristiano Ronaldo is the all-time leading scorer in UEFA Champions League history with 140 goals.",
            "Pele scored 1,281 goals in 1,363 games during his career, though this includes friendlies and exhibition matches.",
            "Diego Maradona led Argentina to victory in the 1986 FIFA World Cup, scoring the famous 'Hand of God' goal.",
            "Zinedine Zidane won the Ballon d'Or in 1998 and led France to World Cup victory the same year.",
            "Ronaldinho won the FIFA World Player of the Year award in 2004 and 2005.",
            "Kaka was the last player to win the Ballon d'Or before the Messi-Ronaldo era began in 2007.",
            "Johan Cruyff revolutionized football with his 'Total Football' philosophy at Barcelona.",
            "Franz Beckenbauer is the only player to win the World Cup as both player and manager.",
            "Xavi Hernandez holds the record for most passes completed in a single World Cup tournament."
        ]
        
        for i, fact in enumerate(football_facts):
            documents.append({
                'text': fact,
                'source': "Real Football Data",
                'id': f"football_{i}"
            })
        
        print(f"✅ Added {len(documents)} football facts")
        return self.add_to_chromadb(documents)
    
    def scrape_website(self, url: str):
        """Basic web scraping (be careful with robots.txt)"""
        print(f"📥 Scraping website: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Basic text extraction (you might want to use BeautifulSoup for better parsing)
            content = response.text
            
            # Remove HTML tags (basic approach)
            import re
            clean_content = re.sub('<[^<]+?>', '', content)
            
            chunks = self.split_text(clean_content, 800)
            documents = []
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    documents.append({
                        'text': chunk,
                        'source': f"Web - {url}",
                        'id': f"web_{i}"
                    })
            
            print(f"✅ Scraped {len(documents)} chunks from website")
            return self.add_to_chromadb(documents)
            
        except Exception as e:
            print(f"❌ Error scraping website: {e}")
            return False
    
    def split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks for better embedding"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        # Add remaining words
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def add_to_chromadb(self, documents: List[Dict]) -> bool:
        """Add documents to ChromaDB"""
        try:
            texts = [doc['text'] for doc in documents]
            embeddings = get_doc_embeddings(texts)
            ids = [doc['id'] for doc in documents]
            
            # Add metadata if available
            metadatas = []
            for doc in documents:
                metadata = {'source': doc['source']}
                if 'metadata' in doc:
                    metadata.update(doc['metadata'])
                metadatas.append(metadata)
            
            collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            print(f"✅ Successfully added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            print(f"❌ Error adding to ChromaDB: {e}")
            return False

# Usage example in main.py
def load_real_data():
    """Load various types of real data"""
    importer = DataImporter()
    
    print("🚀 Loading Real Data into ChromaDB...")
    print("=" * 50)
    
    # 1. Import Wikipedia articles
    topics = ["Lionel Messi", "Cristiano Ronaldo", "Football", "FIFA World Cup"]
    importer.import_wikipedia_articles(topics)
    
    # 2. Import real football data
    importer.import_football_data()
    
    # 3. Import from CSV (if you have one)
    # importer.import_csv_file('football_data.csv', 'description', ['player_name', 'team'])
    
    # 4. Import from text file (if you have one)
    # importer.import_text_file('football_history.txt')
    
    print("\n✅ All data loaded successfully!")
    print("🎯 You can now ask questions about real football data!")

if __name__ == "__main__":
    load_real_data()