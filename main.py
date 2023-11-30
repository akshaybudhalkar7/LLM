
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


pdf_docs = []
os.environ["OPENAI_API_KEY"] = ""


def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(chunks, embeddings)

    return docsearch


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)

    chunks = text_splitter.split_text(text)
    return chunks


def get_pdf_text():
    for f in os.listdir('./datastore'):
        pdf_docs.append(f)
    text = ""
    for pdf in pdf_docs:
        os.chdir(r'D:\LLM\ResumeParser\datastore')
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


if __name__ == '__main__':
    raw_text = get_pdf_text()

    # Get chunks
    text_chunks = get_text_chunks(raw_text)
    print(text_chunks)

    # Create vector store
    vectorstore = get_vectorstore(text_chunks)

    chain = load_qa_chain(OpenAI(), chain_type='stuff')

    query = "Who is Akshay?"

    docs = vectorstore.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    print(response)
