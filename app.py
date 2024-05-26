from pathlib import Path
import pickle
import openai
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import shelve
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

load_dotenv()

class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def __init__(self, openai_api_key, *args, **kwargs):
        super().__init__(openai_api_key=openai_api_key, *args, **kwargs)
        
    def _embed_documents(self, texts):
        return super().embed_documents(texts)

    def __call__(self, input):
        return self._embed_documents(input)

def estimate_complexity(question):
    simple_words = ['name', 'show', 'list', 'tell', 'define', 'what', 'who', 'where']
    difficult_words = ['design', 'experiment', 'compare', 'contrast', 'details', 'theory', 'research', 'evaluate', 'discuss', 'analyze']
    question_lower = question.lower()
    simple_score = sum(question_lower.count(word) for word in simple_words)
    difficult_score = sum(question_lower.count(word) for word in difficult_words)
    complexity_score = difficult_score * 2 - simple_score
    if complexity_score > 3:
        return 5
    elif complexity_score > 0:
        return 5
    else:
        return 1

@app.route('/formulate_response', methods=['POST'])
def formulate_response():
    data = request.json
    prompt = data.get('prompt')
    
    openai_api_key = os.environ["OPENAI_API_KEY"]
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    
    pc = Pinecone(api_key=pinecone_api_key)
    db = pc.Index("pinecone")
    
    chat_history = "\n".join(data.get('chat_history', []))
    prompt_with_history = f"Previous conversation:\n{chat_history}\n\nYour question: {prompt} Answer the question directly."
    
    client = openai.OpenAI(api_key=openai_api_key)
    xq = client.embeddings.create(input=prompt, model="text-embedding-ada-002").data[0].embedding
    
    k = estimate_complexity(prompt)
    results = db.query(vector=[xq], top_k=k, include_metadata=True)
    
    citations = ""
    if results.matches and results.matches[0].score > 0.007:
        model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        db_response = db.query(vector=[xq], top_k=k, include_metadata=True)
        sources = {}
        for match in db_response.matches:
            metadata = match.metadata
            authors = metadata.get('authors', 'Unknown')
            year = metadata.get('year', 'Unknown')
            citation_key = f"({authors.split(',')[0]} et al., {year})"
            if citation_key not in sources:
                sources[citation_key] = (
                    f"\n🦠 {metadata.get('authors', 'Unknown')}\n"
                    f"({metadata.get('year', 'Unknown')}),\n"
                    f"\"{metadata['title']}\",\n"
                    f"PMID: {metadata.get('pub_id', 'N/A')},\n"
                    f"Available at: {metadata.get('url', 'N/A')},\n"
                    f"Accessed on: {datetime.today().strftime('%Y-%m-%d')}\n"
                )
        citations = "\n".join(sources.values())
        query_for_llm = (
            f"Answer directly with extra detail: Question: {prompt_with_history}\n\n"
            f"Cite each sentence with (author, year) in as many sentences as possible, reference all citations. {citations} \n"
            "Do NOT list references."
            f"If the question is about designing an experiment, show step-by-step instructions with immense detail and Cite each sentence with (author, year) in as many sentences as possible, reference all citations. {citations} \n"
        )
        integrated_response = model.predict(query_for_llm)
        response = f"{integrated_response}\n"
    else:
        model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        response = model.predict(prompt_with_history)

    return jsonify({'response': response, 'citations': citations})

if __name__ == '__main__':
    app.run(debug=True)
