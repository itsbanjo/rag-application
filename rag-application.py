import os
import streamlit as st
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from openai import OpenAI
import json
import logging
import re
from typing import List, Dict, Any
from io import StringIO

# Load environment variables
load_dotenv()

# Configure debug mode
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Set up logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

# Create a string buffer to capture log output
log_stream = StringIO()
handler = logging.StreamHandler(log_stream)
handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger.addHandler(handler)

# Elasticsearch configuration
CLOUD_ID = os.getenv("CLOUD_ID")
API_KEY = os.getenv("API_KEY")
ELASTIC_URL = os.getenv("ELASTIC_URL")
INDEX_NAME = os.getenv("INDEX_NAME", "some-index-name")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# App configuration
TITLE = os.getenv("TITLE", "RAG DEMO")
LOGO_PATH = os.getenv("LOGO_PATH", "artefacts/elastic.png")
PROMPT_TEMPLATE_FILE = os.getenv("PROMPT_TEMPLATE_FILE", "artefacts/prompt_template.txt")

# Initialize Elasticsearch client
es = Elasticsearch(
    cloud_id=CLOUD_ID,
    api_key=API_KEY,
)

def display_env_variables():
    """
    Display the environment variables used by the app.
    """
    env_vars = {
        "DEBUG": DEBUG,
        "CLOUD_ID": CLOUD_ID,
        "ELASTIC_URL": ELASTIC_URL,
        "INDEX_NAME": INDEX_NAME,
        "TITLE": TITLE,
        "LOGO_PATH": LOGO_PATH,
        "PROMPT_TEMPLATE_FILE": PROMPT_TEMPLATE_FILE
    }
    
    print("Environment variables used:")
    for key, value in env_vars.items():
        if key in ["API_KEY", "OPENAI_API_KEY"]:
            print(f"{key}: {'*' * 10}")  # Mask sensitive information
        else:
            print(f"{key}: {value}")

def perform_elasticsearch_search(query: str, index: str = INDEX_NAME) -> List[Dict[str, Any]]:
    """
    Perform search using Elasticsearch with the provided query structure.
    """
    es_query = {
        "retriever": {
            "standard": {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "title",
                            "body_content"
                        ]
                    }
                }
            }
        },
        "size": 3
    }
    
    try:
        response = es.search(index=index, body=es_query)
        
        if DEBUG:
            logger.debug(f"Elasticsearch query: {json.dumps(es_query, indent=2)}")
            logger.debug(f"Elasticsearch response: {json.dumps(response, indent=2, default=str)}")
        
        hits = response["hits"]["hits"]
        results = []
        for hit in hits:
            results.append({
                "content": hit["_source"].get("body_content", ""),
                "title": hit["_source"].get("title", "Untitled"),
                "url": hit["_source"].get("url", "No URL provided")
            })
        return results
    except Exception as e:
        logger.error(f"Error in Elasticsearch query: {str(e)}")
        return []

def read_prompt_template(file_path: str) -> str:
    """
    Read the prompt template from a file.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error(f"Prompt template file not found: {file_path}")
        return "Context: {context}\n\nQuestion: {query}\n\nReferences:\n{references}\n\nPlease provide an answer based on the context and cite the references using [1], [2], etc.\n\nAnswer:"

def get_llm_response(context: str, query: str, references: List[Dict[str, str]]) -> str:
    """
    Get a response from OpenAI using the prompt template and include references.
    """
    prompt_template = read_prompt_template(PROMPT_TEMPLATE_FILE)
    
    # Create a string of numbered references
    ref_string = "\n".join([f"[{i+1}] {ref['title']}" for i, ref in enumerate(references)])
    
    prompt = prompt_template.format(
        context=context,
        query=query,
        references=ref_string
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question. If you use information from the references, cite them using [1], [2], etc. Only cite references if you specifically use information from them."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if DEBUG:
            logger.debug(f"OpenAI prompt: {prompt}")
            logger.debug(f"OpenAI response: {json.dumps(response.model_dump(), indent=2)}")
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in OpenAI request: {str(e)}")
        return "Sorry, I encountered an error while processing your request."

def extract_cited_references(response: str, references: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extract only the references cited in the response.
    """
    cited_indices = set(re.findall(r'\[(\d+)\]', response))
    cited_references = []
    for index in cited_indices:
        i = int(index) - 1
        if 0 <= i < len(references):
            cited_references.append(references[i])
    return cited_references

def main():
    # Display environment variables at the start
    display_env_variables()
    
    st.set_page_config(page_title=TITLE, page_icon=LOGO_PATH)
    
    st.image(LOGO_PATH, width=100)
    st.title(TITLE)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Perform Elasticsearch search
            search_results = perform_elasticsearch_search(prompt)
            
            # Prepare context for LLM
            context = "\n".join([result["content"] for result in search_results])
            
            # Get response from LLM
            response = get_llm_response(context, prompt, search_results)
            
            # Extract cited references
            cited_references = extract_cited_references(response, search_results)
            
            # Display the response
            message_placeholder.markdown(response)
            
            # Display only cited references
            if cited_references:
                st.markdown("### References:")
                for i, result in enumerate(cited_references, 1):
                    st.markdown(f"{i}. [{result['title']}]({result['url']})")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    if DEBUG:
        st.write("Debug Information:")
        st.text(log_stream.getvalue())

if __name__ == "__main__":
    main()
