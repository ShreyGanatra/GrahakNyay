from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from datetime import datetime
import os
from prompts import get_prompt


# get current path
current_path = os.path.dirname(os.path.realpath(__file__))

def get_chunks(qa_path):
    path_to_csv = os.path.join(current_path, qa_path)
    loader = CSVLoader(file_path=path_to_csv, 
        metadata_columns=["Filename"],
        source_column="Filename",
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            }
    )
    data = loader.load()
    return data

def get_vectorstore(doc_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={"device": "cuda:0"},
    )
    vectorstore = FAISS.from_documents(documents=doc_chunks, embedding=embeddings)
    return vectorstore

def get_llm():

    # tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    # streamer = TextStreamer(tok,skip_prompt=True)

    # pipe = pipeline(
    #     task="text-generation",
    #     model = model,
    #     tokenizer = tok,
    #     # streamer = streamer,
    #     temperature = 0.0,
    #     pad_token_id=tok.eos_token_id,
    #     device=0,
    #     return_full_text=False,
    #     max_new_tokens=2048,
    #     do_sample=False,
    # )

    # llm = HuggingFacePipeline(
    #     pipeline = pipe,
    #     model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # )

    # llm_engine_hf = ChatHuggingFace(llm=llm, tokenizer=tok)
    # llm = HuggingFaceEndpoint(
    #     endpoint_url="localhost:8080/v1/chat/completions",
    #     streaming=True,
    # )

    # llm_engine_hf = ChatHuggingFace(llm=llm)
    llm_engine_hf = ChatOpenAI(
        # model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        openai_api_key="EMPTY",
        openai_api_base="http://172.17.0.1:8080/v1/",
        temperature=0,
        )
    return llm_engine_hf

def get_conversation_chain(retriever, llm_engine_hf, one_shot=False, general_corpus=False, rag=True):

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "For example:"
        "Chat History: "
        "Human: What is Task Decompostion?"
        "AI: Task Decomposition is the process of breaking down a complex task into smaller and simpler steps. This is achieved through a technique called Chain of Thought (CoT), which instructs the model to \"think step by step\" and utilize more test-time computation to transform big tasks into multiple manageable tasks."
        "Question: What are some of the ways of doing it?"
        "Contextualized Question: What are some of the ways of doing Task Decompositon?"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm_engine_hf, retriever, contextualize_q_prompt
    )
    system_prompt = get_prompt(one_shot=one_shot,general_corpus=general_corpus)

    qa_prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_prompt = qa_prompt.partial(date=datetime.now().strftime("%Y-%m-%d"))
    question_answer_chain = create_stuff_documents_chain(llm_engine_hf, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return  rag_chain


def convert_to_html(text: str) -> str:
    """
    Convert markdown-like text to HTML format.
    Handles newlines, bold, italic, and code formatting.
    """
    replacements = {
        '\n': '<br>',
        '**': '</b>',  # Bold
        '*': '</i>',   # Italic
        '`': '</code>' # Code
    }
    
    # # Handle bold, italic, and code with proper opening tags
    # text = text.replace('**', '<b>', 1)
    # text = text.replace('*', '<i>', 1)
    # text = text.replace('`', '<code>', 1)
    
    # Apply all other replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text