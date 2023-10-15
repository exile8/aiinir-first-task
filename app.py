from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from fastapi import FastAPI
from langchain.schema import messages_to_dict
from langchain.memory.chat_message_histories.redis import RedisChatMessageHistory
from pathlib import Path
import json
from pydantic import BaseModel

# For development
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Preparing vectorstore with documents
pdf_loader = DirectoryLoader('tinkoff-terms', glob='**/*.pdf', loader_cls=PyPDFLoader)
csv_loader = DirectoryLoader('tinkoff-terms', glob='**/*.csv', loader_cls=CSVLoader)

docs = csv_loader.load() + pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(docs)

model_name = 'msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)

llm = LlamaCpp(
    model_path='llama-2-7b-chat.Q4_K_M.gguf',
    temperature=0.75,
    max_tokens=1024,
    top_p=0.1,
    n_ctx=2048,
    callback_manager=callback_manager, 
    verbose=True,
)

PROMPT = '''You are a helpful and honest chatbot-assistant.
    You are having a conversation with a human.
    To answer Question at the end, look at Documents only. Don't use your own knowledge.
    To hold conversation, refer to Chat History.
    Answer only in Russian.
    If you don't know the answer, just say you don't know. Don't make up an answer.
    Your answers must be short and informative.
    Chat History: {chat_history}
    Documents: {context}
    Question: {question}
    Answer: '''

prompt = PromptTemplate(
    template= PROMPT,
    input_variables=['question', 'context', 'chat_history']
)

# window + storage parameters
chat_limit = 5
window_len = 3

chat_storage = {}
chat_dir = 'chat-history/'

class Message(BaseModel):
    message: str
    user_id: str

class Answer(BaseModel):
    answer: str

app = FastAPI()

@app.post('/message')
def message(message: Message):
    memory = chat_storage.get(
        message.user_id,
        ConversationBufferWindowMemory(
            memory_key='chat_history',
            input_key='question',
            k=window_len,
            return_messages=True,
            chat_memory=RedisChatMessageHistory(message.user_id)
        )
    )

    chat = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )

    context = vectorstore.similarity_search_with_relevance_scores(message.message)
    answer = chat.predict(question=message.message, context=context)

    if len(memory.chat_memory.messages) > chat_limit * 2:
        user_file = Path(chat_dir + message.user_id + '.json')
        if not user_file.is_file():
            user_file.touch()
        with user_file.open('w', encoding='utf-8') as ufp:
            # dump to persist
            json.dump(messages_to_dict(memory.chat_memory.messages), ufp, ensure_ascii=False)
            # restore current window
            window = memory.chat_memory.messages[-(2 * window_len):]
            memory.chat_memory.clear()
            for i in range(window_len):
                memory.chat_memory.add_message(window.pop(0))
                memory.chat_memory.add_message(window.pop(0))

    return Answer(answer=answer)