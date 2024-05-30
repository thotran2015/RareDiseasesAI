from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# need to do ollama pull llama3
ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3"
)
# print(ollama.invoke("why is the sky blue"))

# loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
loader = DirectoryLoader("../HPP_data", "*.txt", use_multithreading=True, show_progress=True)
data = loader.load()
print("Number of docs: ", len(data))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
print("Number of splits: ", len(all_splits))
# need to do ollama pull nomic-embed-text and pip install chromadb
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits[:300], embedding=oembed)

# question = "What is Hypophosphatasia(HPP)?"
# docs = vectorstore.similarity_search(question)
# print("docs", len(docs))

qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

while True:
    question = input("Ask a question: ")
    if question == "exit":
        break
    print(qachain.invoke({"query": question}))
