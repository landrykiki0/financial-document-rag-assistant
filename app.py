import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def get_conversation(vstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = vstore.as_retriever(search_kwargs={"k": 2})
    def get_context(inputs):
        docs = retriever.invoke(inputs["input"])
        return "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Réponds en utilisant ce contexte:\n\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    chain = (
        RunnablePassthrough.assign(context=get_context)
        | prompt
        | llm
        | StrOutputParser()
    )
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    return chain_with_memory

def get_vectors(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_chunks(text):
    text_splitter =  RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators=["\n\n", "\n", " ", ""]
        )
    chunks = text_splitter.split_text(text)
    return chunks

def get_text_from_pdf(docs):
    text = ""
    for doc in docs:
        single_pdf = PdfReader(doc)
        for page in single_pdf.pages:
            text+= page.extract_text()
        text += "fin du document"
    return text

def main():
    st.set_page_config(page_title="RAG Application", page_icon=":books:")
    st.header("Financial Document Assistant :books:")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Pose ta question ici")
    if question:
        st.session_state.messages.append({"role":"user", "content":question})
        with st.chat_message("user"):
            st.write(question)
        if "conversation" in st.session_state:
            with st.spinner("reflexion......"):
                response = st.session_state.conversation.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": "user_1"}}
                )
                st.session_state.messages.append({"role":"assistant", "content":response})
                with st.chat_message("assistant"):
                    st.write(response)

    with st.sidebar:
        st.subheader("Tes documents")
        docs = st.file_uploader("Charge tes documents et clique sur téléverser", accept_multiple_files=True)
        if st.button("Téléverser"):
            with st.spinner("Chargement...."):
                raw_text = get_text_from_pdf(docs)
                chunks = get_chunks(raw_text)
                vstore = get_vectors(chunks)
                st.write("Done")
                st.session_state.conversation = get_conversation(vstore)

if __name__ == '__main__':
    main()