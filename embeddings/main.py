
import asyncio
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings()

# Streamlit UI setup
st.set_page_config(page_title="LangChain Chatbot")
st.title("🤖 LangChain Chatbot")
user_input = st.text_input("Ask a question:", "")


async def process_file(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

# Correct Windows path (use raw string or double backslashes)
file_path = r"C:\Users\debaj\OneDrive\Documents\All_Python\All_projects\Generative_AI\OpenAI_API_For_Python_developers\embeddings\sample_pdf\NVDA 2QFY24.pdf"

# Run the async function
docs = asyncio.run(process_file(file_path))

# Print results
# print(len(docs))
# print(f"{docs[0].metadata}\n")
# print(docs[0].page_content)

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)

split_docs = splitter.split_documents(docs)
# print(split_docs)

model = ChatOpenAI(
    model="gpt-4",
    streaming=False,
    temperature=0.9,
    # openai_api_key=os.getenv("OPENAI_API_KEY") # this is not required if u save the key in a environment variable
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are Chainlit GPT, a helpful assistant."),
    # HumanMessagePromptTemplate.from_template("{question}")
    HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion:\n{question}")
])

db = Chroma.from_documents(split_docs, embeddings)
print(db)

# query it
# docs = db.similarity_search(user_input)
docs = db.similarity_search(user_input, k=4)  # Fetch top 4 chunks

# return results

context = "\n\n".join([doc.page_content for doc in docs]) 

chain = prompt | model | StrOutputParser()



with st.spinner("Thinking..."):
    response = chain.invoke({"question": user_input, "context": context}) # same as response = chain.invoke(formatted_prompt)
    st.success(response)