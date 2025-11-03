import re
from flask import Flask, request, jsonify, render_template, Response, stream_with_context,url_for, redirect, url_for, session, flash
from dotenv import load_dotenv
from flask_cors import CORS 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import sys
import os
import json
import pandas as pd
from langchain_core.messages import AIMessage
from datetime import datetime
from utils import get_chunks, get_vectorstore, get_llm, get_conversation_chain, convert_to_html
import uuid


from utils import load_sectoral_docs, load_few_shot_examples
from prompts import get_prompt


BASE_URL = '/consumer_chatbot'
all_rag_vectorstore = None
sector_rag_vectorstore = None
all_rag_history = {}
sector_rag_history = {}
no_rag_history = {}
all_rag_conversation_chain = None
sector_rag_conversation_chain = None
no_rag_conversation_chain = None
session_sector_map = {}         # session_id -> sector
session_chat_count = {}         # session_id -> number of messages
session_model_map = {}          # session_id -> model_name
sector_retriever = {}



def get_all_rag_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in all_rag_history:
        all_rag_history[session_id] = ChatMessageHistory()
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        all_rag_history[session_id].add_message(AIMessage(content=initial_message))
    return all_rag_history[session_id]

def get_sector_rag_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in sector_rag_history:
        sector_rag_history[session_id] = ChatMessageHistory()
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        sector_rag_history[session_id].add_message(AIMessage(content=initial_message))
    return sector_rag_history[session_id]

def get_no_rag_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in no_rag_history:
        no_rag_history[session_id] = ChatMessageHistory()
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        no_rag_history[session_id].add_message(AIMessage(content=initial_message))
    return no_rag_history[session_id]


def initialize_app():
    global all_vectorstore, all_retriever, sector_retriever
    global sector_docs_map, few_shot_map

    print("Initializing application data...")

    # Load the main corpus CSV into a pandas DataFrame once
    corpus_path = "final_corpus.csv"
    df = pd.read_csv(corpus_path)
    # df['normalized_file'] = df['File'].str.replace(r'[\s_]+', '', regex=True).str.lower()

    # --- Create the general retriever for the initial conversation ---
    all_rag_chunks = get_chunks(corpus_path)
    all_vectorstore = get_vectorstore(all_rag_chunks)
    all_retriever = all_vectorstore.as_retriever()

    # --- Create a specialized retriever for each sector ---
    sector_docs_map = load_sectoral_docs()
    few_shot_map = load_few_shot_examples()
    os.makedirs("Sector_CSV", exist_ok=True)

    print("Creating sector-specific retrievers...")
    for sector in sector_docs_map.keys():
        print(f"  - Processing sector: {sector}")
        
        # normalized_sector = re.sub(r'[\s_]+', '', sector).lower()
        # print(f"normalized_sector_name: {normalized_sector}\n")
        
        # 3. Filter using the normalized columns. This will always work.
        # We still include 'General' from the original 'File' column.
        filtered_df = df[
            (df['File'] == sector) | 
            (df['File'] == 'General')
        ].copy()
        
        filtered_df.to_csv(f"Sector_CSV/{sector}.csv")

        # Convert the filtered DataFrame rows into LangChain Document objects
        sector_docs = get_chunks(f"Sector_CSV/{sector}.csv")
        # Create and store the vector store and retriever for this sector
        if sector_docs:
            sector_vectorstore = get_vectorstore(sector_docs)
            sector_retriever[sector] = sector_vectorstore.as_retriever()
        else:
            print(f"    - Warning: No documents found for sector '{sector}'. Using general retriever as fallback.")
            sector_retriever[sector] = all_retriever # Fallback to general retriever

    print("All documents and retrievers loaded.")



# def initialize_app():
#     global all_vectorstore, all_retriever
#     global sector_docs_map, few_shot_map

#     print("Initializing application data...")



#     all_rag_chunks = get_chunks("final_corpus.csv")
#     all_vectorstore = get_vectorstore(all_rag_chunks)
#     all_retriever = all_vectorstore.as_retriever()

#     # Load sectoral data
#     sector_docs_map = load_sectoral_docs()
#     few_shot_map = load_few_shot_examples()



#     for sector in sector_docs_map:
#         sector_data = sector_data["File"=="General" && "f{File.replace(" ", "_")}.docx"==f"{sector}"] # File is the column name from which we wanna fetch
#         sector_data.to_csv(f"{sector}.csv")

#         sector_rag_chunks = get_chunks(f"{sector}.csv")
#         sector_vectorstore = get_vectorstore(sector_rag_chunks)
#         sector_retriever[sector] = sector_vectorstore.as_retriever()
    

#     print("All documents loaded.")




app = Flask(__name__, static_url_path='/consumer_chatbot/static')
CORS(app)
app.secret_key = os.urandom(24)
load_dotenv()

with app.app_context():
    initialize_app()



@app.route(f'{BASE_URL}/reset_session', methods=['POST'])
def reset_session():
    data = request.json
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400

    # Remove from all history stores
    all_rag_history.pop(session_id, None)
    sector_rag_history.pop(session_id, None)
    no_rag_history.pop(session_id, None)

    # Remove sector classification state
    session_sector_map.pop(session_id, None)
    session_chat_count.pop(session_id, None)

    return jsonify({"message": f"Session {session_id} has been reset."})


@app.route(f'{BASE_URL}/get_models', methods=['GET'])
def get_models():
    # Static list of available Groq + hosted models
    available_models = [
        "hosted",
        "groq-llama3-8b-8192",
        # "groq-llama3-70b-8192",
        # "groq-mixtral-8x7b-32768",
        # "groq-gemma-7b-it",
        # # "gemini-2.5-pro",
        # "groq-openai/gpt-oss-120b",
        # "groq-openai/gpt-oss-20b",
        # "groq-llama-4-scout-17b-16e-instruct",
        "hf-model-openai/gpt-oss-20b:fireworks-ai",
        "gemini-2.5-flash",
        "gemini-2.5-pro"
    ]
    return jsonify({"models": available_models})


@app.route(f'{BASE_URL}/set_model', methods=['POST'])
def set_model():
    data = request.json
    model = data.get("model")
    session_id = data.get("session_id")

    if not model or not session_id:
        return jsonify({"error": "Model and session_id are required"}), 400

    session_model_map[session_id] = model
    return jsonify({"message": f"Model set to {model} for session {session_id}"})



@app.route(f'/{BASE_URL}')
def index():
    return render_template('index.html', BASE_URL=BASE_URL)

@app.route(f'{BASE_URL}/terms_of_use')
def terms_of_use():
    return render_template('terms_of_use.html', BASE_URL=BASE_URL)

@app.route(f'{BASE_URL}/get_session_id', methods=['GET'])
def get_session_id():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})

@app.route(f'/{BASE_URL}/initial_message', methods=['GET'])
def initial_message():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    history = get_all_rag_session_history(session_id)
    initial_ai_message = history.messages[0].content if history.messages else "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
    return jsonify({"message": initial_ai_message})


@app.route(f'{BASE_URL}/chat', methods=['POST', 'GET'])
def consumer_chat():
    return chat_handler(all_rag_conversation_chain, get_all_rag_session_history, rag=True)


def sector_prompt_for(session_id):
    sector = session_sector_map[session_id]
    docs = sector_docs_map[sector]
    def escape_curly_braces(text):
        return text.replace("{", "{{").replace("}", "}}")

    sector_text = "\n\n".join(escape_curly_braces(doc.page_content) for doc in docs)

    sector_prompt = get_prompt(
        one_shot=False,
        general_corpus=False,
        # sector_content=sector_text,
        sector_content=None,
        custom_one_shot_example=few_shot_map.get(sector, ""),
        sector=sector
    )

    return sector_prompt


def chat_handler(conversation_chain, get_session_history_func, rag):
    if request.method == 'POST':
        data = request.json
    else:  
        data = request.args
    user_input = data.get('message')
    session_id = data.get('session_id')
    
    # Get model for this session (default to hosted if not set)
    model_name = session_model_map.get(session_id, "hosted")
    llm_engine_hf = get_llm(model=model_name)

    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    if rag:
        # Track message count
        session_chat_count[session_id] = session_chat_count.get(session_id, 0) + 1

        print("session_chat_count: ", session_chat_count)

        # Sector classification after second message
        if session_id not in session_sector_map and session_chat_count[session_id] == 2:
            messages = get_all_rag_session_history(session_id).messages
            first_user_message = next((m.content for m in messages if m.type == 'human'), '')
            second_user_message = user_input
            classification_prompt = f"""
            You are a classification assistant. Given the following two user messages, classify them into one of the following sectors:
            {', '.join(sector_docs_map.keys())}

            Message 1: {first_user_message}
            Message 2: {second_user_message}

            Respond with only the sector name that best represents the combined content of the messages.
            """
            result = llm_engine_hf.invoke(classification_prompt)
            predicted_sector = result.content.strip()
            print(f"\nPredicted Sector: {predicted_sector}\n")
            if predicted_sector in list(sector_docs_map.keys()):
                session_sector_map[session_id] = predicted_sector

                # =========================================================
                # ==              HISTORY TRANSFER LOGIC                 ==
                # =========================================================
                print(f"Switching to sector-specific chain for '{predicted_sector}'. Transferring chat history...")
                # Get the existing general history
                general_history = get_all_rag_session_history(session_id)
                # Get the new (and currently empty) sector history
                sector_history = get_sector_rag_session_history(session_id)
                
                # Clear the new history (which only has an initial message)
                sector_history.clear() 
                # Copy all messages from the old history to the new one
                sector_history.add_messages(general_history.messages)
                # =======================================================
                # ==                       END                         ==
                # =======================================================

        # Select conversation chain
        if session_id in session_sector_map:
            sector = session_sector_map[session_id]
            chain = RunnableWithMessageHistory(
                get_conversation_chain(sector_retriever[sector], llm_engine_hf, sector_prompt=sector_prompt_for(session_id)),
                get_sector_rag_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        else:
            chain = RunnableWithMessageHistory(
                get_conversation_chain(all_retriever, llm_engine_hf),
                get_all_rag_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

        config = {"configurable": {"session_id": session_id}}
        output = {}
        def generate():
            buffer = []
            buffer_size = 5
            curr_key = None
            partial_link = ""
            try:
                for chunk in chain.stream({"input": user_input}, config):
                    for key in chunk:
                        if key not in output:
                            output[key] = chunk[key]
                        else:
                            output[key] += chunk[key]
                        # if key != curr_key:
                        #     print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
                        # else:
                        #     print(chunk[key], end="", flush=True)
                        curr_key = key

                        if key == "answer":
                            buffer.append(chunk[key])
                            if len(buffer) >= buffer_size:
                                chunk = ''.join(buffer)
                                buffer.clear()
                                # Handle partial links
                                if partial_link:
                                    chunk = partial_link + chunk
                                    partial_link = ""
                                if chunk.count('[') > chunk.count(')'):
                                    partial_link = chunk[chunk.rfind('['):]
                                    chunk = chunk[:chunk.rfind('[')]
                                chunk = convert_to_html(chunk)
                                yield f"data: {chunk}\n\n"
                if buffer:
                    chunk = ''.join(buffer)
                    if partial_link:
                        chunk = partial_link + chunk
                    chunk = convert_to_html(chunk)
                    yield f"data: {chunk}\n\n"

                # Send context data as separate event
                if 'context' in output:
                    context_data = [doc.page_content for doc in output['context']]
                    yield f"event: context\ndata: {json.dumps(context_data)}\n\n"

                yield "event: progress\ndata: \n\n"

            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
                yield "event: done\ndata: \n\n"
        
        return Response(
                stream_with_context(generate()),
                content_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
    else:
        chain = RunnableWithMessageHistory(
            conversation_chain,
            get_session_history_func,
            input_messages_key="input",
            history_messages_key="chat_history",
        ) 


    
@app.route(f'{BASE_URL}/get_chat_history', methods=['GET'])
def get_consumer_chat_history():
    return get_chat_history(get_all_rag_session_history)


def get_chat_history(get_session_history_func):
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    history = get_session_history_func(session_id)
    # print("history",history)
    chat_history = [
        {"role": "AI" if isinstance(msg, AIMessage) else "Human", "content": msg.content}
        for msg in history.messages
    ]
    return jsonify({"chat_history": chat_history})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=50002)