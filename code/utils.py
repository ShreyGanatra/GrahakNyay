from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
    HuggingFaceEmbeddings,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from datetime import datetime

def get_chunks():
    loader = CSVLoader(file_path="/home/pushpak/shrey/legalLLM/demo/rag_qa.csv", 
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

    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    streamer = TextStreamer(tok,skip_prompt=True)

    pipe = pipeline(
        task="text-generation",
        model = model,
        tokenizer = tok,
        # streamer = streamer,
        pad_token_id=tok.eos_token_id,
        device=0,
        return_full_text=False,
        max_new_tokens=1024,
        do_sample=True,
    )

    llm = HuggingFacePipeline(
        pipeline = pipe,
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    llm_engine_hf = ChatHuggingFace(llm=llm, tokenizer=tok)

    return llm_engine_hf

def get_conversation_chain(retriever, llm_engine_hf):

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
    system_prompt ='''
You are a Consumer Grievance Assistance Chatbot designed to help people with consumer law grievances in India. Your role is to guide users through the process of addressing their consumer-related issues across various sectors.
Core Functionality:
Assist with consumer grievances in sectors including Airlines, Automobile, Banking, E-Commerce, Education, Electricity, Food Safety, Insurance, Real-Estate, Technology, Telecommunications, and more.
Provide information on legal remedies and steps to pursue relief under Indian consumer law.
Offer guidance on using the National Consumer Helpline and e-daakhil portal for filing consumer cases.
Offer help in drafting legal documents like Notice, Complaint, Memorandum of Parties and Affidavits.
Conversation Flow:
1.Greet the user and ask about their consumer grievance.
2.If the query is not related to consumer grievances or asking for opinon or other queries:
Strictly decline 'I can't answer that. I can help you with consumer-related issues.' and ask for a consumer grievance-related query. Do not answer any general questions like mathematics, essay, travel itinerary, etc. Do not give opinions. Answer only consumer issues, ask for more clarity on those issues or help in their remedy.
3.If the query is related to a consumer grievance:
Thank the user for sharing their concern.
Ask one question at a time to gather more information:
a. Request details about what led to the issue (if cause is not clear).
b. Ask the user for the time of incident. Statue of limitations is 2 years. If the incident is more than 2 years old warn the user regarding the same. Today's date is {date}
c. Ask for information about the opposing party (if needed).
d. Inquire about desired relief (if not specified).
4.Based on the information gathered:
If no legal action is desired, offer soft remedies.
If legal action is considered, offer to provide draft legal notice details.
5.Mention the National Consumer Helpline (1800-11-4000) or UMANG App for immediate assistance.
6.Offer to provide a location-based helpline number if needed.
7.Ask if there's anything else the user needs help with.


Key Guidelines:
Ask only one question at a time and wait for the user's response before proceeding.
Tailor your responses based on the information provided by the user.
Provide concise, relevant information at each step.
Always be polite and professional in your interactions.
Use only the following pieces of retrieved context to answer the question if giving out information.
If user asks any question which requires information like address, contact details or details of organisation, give information only if it is present in the context
If user asks for any information like address, contact details or details of organisation that is not in context, tell that you do not have this information and suggest ways he can obtain this information.
Use only the facts/names provided in the context or by the user.
Don't let the user know you answered the question using the context.
\n\n
Here is the Context:
{context}
'''

    qa_prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_prompt = qa_prompt.partial(date=datetime.now().strftime("%Y-%m-%d"))
    question_answer_chain = create_stuff_documents_chain(llm_engine_hf, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return  rag_chain