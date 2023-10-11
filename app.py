from langchain.llms import LlamaCpp
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.5,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful, respectful and honest assistant. \
            Always answer as helpfully as possible, while being safe. \
            Your answers should not include any harmful, unethical, \
            racist, sexist, toxic, dangerous, or illegal content. \
            Please ensure that your responses are socially unbiased and positive in nature. \
            If a question does not make any sense, or is not factually coherent, \
            explain why instead of answering something not correct. \
            If you don't know the answer to a question, please don't share false information."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

chat = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

while True:
    question = input()
    chat({"question": question})
