# Bot_Knowledge
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from io import BytesIO
import boto3
from PyPDF2 import PdfReader
from langchain.schema import Document
import json
import requests
 
st.title("Knowledge Management Chatbot")
 
# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
 
aws_access_key = ''
aws_secret_key = ''
region_name = 'us-east-1'
 
# Initialize S3 and Bedrock clients
s3 = boto3.client('s3', region_name=region_name, aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
client = boto3.client('bedrock-runtime', region_name=region_name, aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
 
bucket_name = 'docsummhhfl'
object_key = 'HR/policy/hrpolicytest.pdf'
 
st.write(f"Attempting to access file in Bucket: {bucket_name}, Key: {object_key}")
 
# Function to stream file from S3
def stream_file_from_s3(bucket_name, object_key):
    try:
        file_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        return BytesIO(file_obj['Body'].read())
    except Exception as e:
        st.error("Error streaming file from S3")
        return None
 
# Load and process the PDF file
pdf_file = stream_file_from_s3(bucket_name, object_key)
 
if pdf_file:
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    docs = []
 
    for page in reader.pages:
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text))
 
    st.success("Loaded")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=15)
    combined_text = "".join([doc.page_content for doc in docs])
    split_texts = text_splitter.split_text(combined_text)
    documents = [Document(page_content=text) for text in split_texts]
 
    hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.from_documents(documents, hf_embedding)
 
    # Function to call AWS Bedrock LLM and return the response
    def call_bedrock_llm(prompt):
        try:
            # Prepare the body content for the request
            body_content = {
                "prompt": prompt,
                "max_gen_len": 512,
                "temperature": 0.5,
                "top_p": 0.9
            }
 
            # Convert body to JSON format
            body_json = json.dumps(body_content)
 
            # Make the request to the Bedrock LLM
            response = client.invoke_model(
                modelId="meta.llama3-70b-instruct-v1:0",  # Model ID
                body=body_json,
                contentType="application/json",
                accept="application/json"
            )
 
            # Read and process the response
            response_body = response['body'].read().decode('utf-8')
            result = json.loads(response_body)
 
            # Extract the 'generation' field
            if 'generation' in result:
                generation_text = result['generation']
                return generation_text
            else:
                st.error("No 'generation' field found in response. Full response: {}".format(result))
                return None
 
        except Exception as e:
            st.error(f"Error calling AWS LLM: {e}")
            return None
 
    # Function to translate the bot's response using Bhashini API
    def translate_response(response_text, target_language):
        api_url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getmodelsPipeline"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "426d392042-9028-4f13-aea7-ad172f8048f8"
        }
        data = {
            "source_text": response_text,
            "source_language": "en",
            "target_language": target_language
        }
        try:
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                translated_text = response.json().get("translated_text", response_text)
                return translated_text
            else:
                st.error(f"Error in translation: {response.status_code}")
                return response_text
        except Exception as e:
            st.error(f"Error calling Bhashini API: {e}")
            return response_text
 
    # Language mapping for translation
    language_mapping = {
        "English": "en",
        "Kashmiri": "ks",
        "Nepali": "ne",
        "Bengali": "bn",
        "Marathi": "mr",
        "Sindhi": "sd",
        "Telugu": "te",
        "Gujarati": "gu",
        "Gom": "gom",
        "Urdu": "ur",
        "Santali": "sat",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Manipuri": "mni",
        "Tamil": "ta",
        "Hindi": "hi",
        "Punjabi": "pa",
        "Odia": "or",
        "Dogri": "doi",
        "Assamese": "as",
        "Sanskrit": "sa",
        "Bodo": "brx",
        "Maithili": "mai"
    }
 
    # Select language for bot response
    language = st.selectbox(
        "Choose the language for the bot's response",
        options=list(language_mapping.keys())
    )
 
    # Define the prompt template
    prompt_template = """
    You are a Knowledge Management specialist. Answer the queries from an expert perspective.
    Answer the following questions based on the provided document and chat history.
 
    Context:
    {context}
 
    Conversation History:
    {chat_history}
 
    Question: {user_question}
    """
 
    # Create the document chain with the custom Bedrock LLM
    document_chain = create_stuff_documents_chain(llm=call_bedrock_llm, prompt=ChatPromptTemplate.from_template(prompt_template))
    retriever = vector_db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
 
    # User input for the question
    user_question = st.text_input("Ask a question about the document", key="input")
 
    if user_question:
        # Prepare conversation history
        conversation_history = ""
        for chat in st.session_state['chat_history']:
            conversation_history += f"You: {chat['user']}\nBot: {chat['bot']}\n"
 
        # Format the prompt
        prompt = prompt_template.format(context=combined_text, chat_history=conversation_history, user_question=user_question)
        response_text = call_bedrock_llm(prompt)
 
        # Translate response if needed
        if response_text:
            if language != "English":
                response_text = translate_response(response_text, language_mapping[language])
 
            # Append to chat history
            st.session_state.chat_history.append({"user": user_question, "bot": response_text})
 
    # Display conversation history
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #DCF8C6;'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #ECECEC; margin-top: 5px;'><strong>Bot:</strong> {chat['bot']}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
