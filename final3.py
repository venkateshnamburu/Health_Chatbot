import streamlit as st
import boto3
import fitz  # PyMuPDF
import google.generativeai as genai
import json
import io
import re

# --- CONFIG ---
GEMINI_API_KEY = "AIzaSyAwlJ-6BfIS3HeMPsjYU9Pc0A89lgykjuw"
BUCKET_NAME = "heathrecords"
AWS_REGION = "us-east-1"

# --- SETUP ---
genai.configure(api_key=GEMINI_API_KEY)
s3 = boto3.client("s3", region_name=AWS_REGION)

# --- Extract PDF Text ---
def extract_text_from_pdf_stream(pdf_stream):
    doc = fitz.open("pdf", pdf_stream)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Ask Gemini ---
def ask_gemini_over_documents(docs_texts, question):
    combined_docs = "\n\n---\n\n".join(
        [f"Document: {name}\n{text[:4000]}" for name, text in docs_texts.items()]
    )

    prompt = f"""
You are a smart document understanding assistant.

Here are multiple documents from which you may need to answer a question. Each document is separated by "---".

{combined_docs}

Now answer this question: "{question}"

Return ONLY a valid JSON object with your answer. Do not include explanations, markdown, or any extra text.

Example format:
{{
  "answer": "...",
  "source_document": "filename.pdf",
  "confidence": "High/Medium/Low",
  "relevant_snippet": "..."
}}
"""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Clean Gemini Output ---
def clean_json_text(text):
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE)
    return cleaned.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Ask Any Document from S3", layout="wide")
st.title("📄💬 Ask a Question — I’ll Find the Answer from Any PDF in S3")

with st.spinner("🔍 Loading and reading all documents..."):
    try:
        docs_texts = {}
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        pdf_keys = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".pdf")]

        for key in pdf_keys:
            obj_response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            file_stream = obj_response["Body"].read()
            text = extract_text_from_pdf_stream(file_stream)
            docs_texts[key] = text
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        st.stop()

question = st.text_input("❓ Ask your question:")

if question:
    with st.spinner("🤖 Thinking..."):
        raw_response = ask_gemini_over_documents(docs_texts, question)
        cleaned = clean_json_text(raw_response)

        try:
            json_output = json.loads(cleaned)
            st.success("✅ Answer:")
            st.json(json_output)
        except json.JSONDecodeError:
            st.error("❌ Could not parse response as JSON.")
            st.code(raw_response)
