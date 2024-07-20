import streamlit as st
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import openai
import os
import dotenv

dotenv.load_dotenv()

# openai_api_key = os.getenv('OPEN_AI_KEY')

# if not openai_api_key:
#     raise ValueError("No OpenAI API key found in environment variables. Please set the 'OPENAI_API_KEY' environment variable.")
# openai.api_key = openai_api_key
#
# # Load PDF and process
pdf_loader = PyPDFLoader(file_path="emaild(1).pdf")
docs = pdf_loader.load()
# print(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(api_key='sk-proj-KB64iQAlENuFceKD32iaT3BlbkFJSGnSkcABi8TBjxJCWU7d')
# print(embeddings)
llm = ChatOpenAI(model="gpt-4o",api_key='sk-proj-KB64iQAlENuFceKD32iaT3BlbkFJSGnSkcABi8TBjxJCWU7d')
# print(llm.invoke("what is open ai"))
vector = Chroma.from_documents(splits,embeddings)
retriever = vector.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """
system You are a top {content_type} writer.
user Write a {content_type} keeping in mind that, {company_name}'s {product} is the perfect solution for {painpoint} that makes it a must-have for school owners due to the following benefits: {benefits}. You are an expert marketer and {role} whose message no one can ignore. Write a highly impactful {content_type} that highlights the product/service and encourages customers to read the full content. The introductory paragraph should be short and precise, followed by a list of pointers and a short and actionable conclusion note. Write a compelling call to action which directs the user to click on a link to the company's website: {company_link} that takes them to a project that Epack has done in the past in the same field. Make it relatable with {target_audience} and highlight their pain point effectively by giving them practical solutions rather than just praising their own offering. Avoid generic salesy language that no one will read till the end. Keep it short and give valuable information to the relevant user. Invoke the emotion of {emotion}. Be very creative. Keep in mind the tone which is {tone}. Keep the length {length} words. use this

Context: {context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=[
        "context",
        "content_type",
        "company_name",
        "product",
        "painpoint",
        "benefits",
        "role",
        "target_audience",
        "company_link",
        "emotion",
        "tone",
        "length"
    ],
)

# Streamlit UI
st.title("AI Content Generator")



content_type = st.text_input("Content Type", value="email writer")
company_name = st.text_input("Company Name", value="efeb")
product = st.text_input("Product", value="construction")
painpoint = st.text_input("Pain Point", value="cost")
benefits = st.text_input("Benefits", value="saving")
role = st.text_input("Role", value="email writer")
target_audience = st.text_input("Target Audience", value="school")
company_link = st.text_input("Company Link", value="epack.in")
emotion = st.text_input("Emotion", value="happy")
tone = st.text_input("Tone", value="consultive")
length = st.number_input("Length", value=100)

generate = st.button("Generate Content")

if generate:
    rag_chain = prompt | llm | StrOutputParser()
    context = retriever.invoke(painpoint)
    formatted_context = format_docs(context)
    response = rag_chain.invoke({"context":formatted_context, "content_type":content_type, "company_name":company_name, "product":product, "painpoint":painpoint, "benefits":benefits, "role":role, "target_audience":target_audience, "company_link":company_link, "emotion":emotion, "tone":tone, "length":length})
    st.markdown("## Generated Content")
    st.write(response)
