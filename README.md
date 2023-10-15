# aiinir-first-task
A Q&A service powered by Llama2-7B

## Run
1. Make sure your working directory is **aiinir-first-task**.
2. Download [llama](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf) and move it into the working directory.
3. Run the following commands:
```
docker-compose build
```

```
docker-compose up
```

## Implementation notes
* Current implementation is a toy-app with limitations due to development needs, Parameters window_len and chat_limit can be altered (currently 3 and 5 respectively).
* QA search over ChromaDB.
* Chat history session storage in RedisDB.
* Persistent storage of chat history per user in **chat-history** directory.
* QA database in **tinkoff-terms** directory. 
* [Chat model](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF) (Q4_K_M).
* [Embeddings model](https://huggingface.co/sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned).