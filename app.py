from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders import (
    UnstructuredPDFLoader,
    CSVLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from fastapi import FastAPI

# For development
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

pdf_loader = DirectoryLoader('tinkoff-terms', glob='**/*.pdf', loader_cls=UnstructuredPDFLoader)
csv_loader = DirectoryLoader('tinkoff-terms', glob='**/*.csv', loader_cls=CSVLoader)

docs = csv_loader.load() + pdf_loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(docs)

# TODO: experiment with embeddings model
model_name = 'multi-qa-distilbert-cos-v1'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)

# TODO: Calibrate precision and randomization
llm = LlamaCpp(
    model_path='llama-2-7b-chat.Q4_K_M.gguf',
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    n_ctx=1024,
    callback_manager=callback_manager, 
    verbose=True,
)

# TODO: experiment with prompt language
prompt = PromptTemplate(
    template= '''You are a helpful and honest chatbot-assistant.
    You are having a conversation with a human.
    To answer questions, refer to context.
    To hold conversation, refer to chat history.
    Answer only in Russian.
    If you don't know the answer, just say you don't know. Don't make up an answer.
    Chat History: {chat_history}
    Context: {context}
    Question: {question}
    Answer: ''',
    input_variables=['question', 'context', 'chat_history']
)


memory = ConversationBufferMemory(memory_key='chat_history', input_key='question', return_messages=True)

chat = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

app = FastAPI()

@app.post('/message')
def message(user_id : str, message : str):
    context = vectorstore.max_marginal_relevance_search(message)
    answer = chat.predict(question=message, context=context)

    return {'user_id': user_id, 'answer': answer}