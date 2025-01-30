from flask import Flask, request, jsonify, render_template, Response, stream_with_context,url_for, redirect, url_for, session, flash
from dotenv import load_dotenv
from flask_cors import CORS 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import sys
import os
import json
from langchain_core.messages import AIMessage
from datetime import datetime
from utils import get_chunks, get_vectorstore, get_llm, get_conversation_chain, convert_to_html
import uuid


BASE_URL = '/consumer_chatbot'
all_rag_vectorstore = None
sector_rag_vectorstore = None
all_rag_history = {}
sector_rag_history = {}
no_rag_history = {}
all_rag_conversation_chain = None
sector_rag_conversation_chain = None
no_rag_conversation_chain = None


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
    global all_vectorstore,sector_vectorstore,all_rag_conversation_chain,sector_rag_conversation_chain
    print("Initializing application...")
    all_rag_chunks = get_chunks("rag_qa.csv")
    all_vectorstore = get_vectorstore(all_rag_chunks)
    all_retriever = all_vectorstore.as_retriever()
    # sector_chunks = get_chunks("sector_qa.csv")
    # sector_vectorstore = get_vectorstore(sector_chunks)
    # sector_retriever = sector_vectorstore.as_retriever()
    print("Documents processed successfully")
    llm_engine_hf = get_llm()
    all_rag_conversation_chain = get_conversation_chain(all_retriever,llm_engine_hf,one_shot=True,general_corpus=True)
    # sector_rag_conversation_chain = get_conversation_chain(sector_retriever,llm_engine_hf,one_shot=False,general_corpus=True)
    # no_rag_conversation_chain = get_conversation_chain(all_retriever,llm_engine_hf,one_shot=True,general_corpus=False)
    
    print("Conversation chain initialized")

app = Flask(__name__, static_url_path='/consumer_chatbot/static')
CORS(app)
app.secret_key = os.urandom(24)
load_dotenv()

with app.app_context():
    initialize_app()

@app.route(f'/{BASE_URL}')
def index():
    return render_template('index.html', BASE_URL=BASE_URL)

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


def chat_handler(conversation_chain, get_session_history_func, rag):
    if request.method == 'POST':
        data = request.json
    else:  
        data = request.args
    user_input = data.get('message')
    session_id = data.get('session_id')
    print(datetime.now())
    print("history", get_session_history_func(session_id))
    print("user:", user_input)
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    if rag:
        conversational_rag_chain = RunnableWithMessageHistory(
            conversation_chain,
            get_session_history_func,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        ) 
        chain = conversational_rag_chain 
        config = {"configurable": {"session_id": session_id}}
        output = {}
        def generate():
            buffer = []
            buffer_size = 5
            curr_key = None
            try:
                for chunk in chain.stream({"input": user_input}, config):
                    for key in chunk:
                        if key not in output:
                            output[key] = chunk[key]
                        else:
                            output[key] += chunk[key]
                        if key != curr_key:
                            print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
                        else:
                            print(chunk[key], end="", flush=True)
                        curr_key = key

                        if key == "answer":
                            buffer.append(chunk[key])
                            if len(buffer) >= buffer_size:
                                chunk = ''.join(buffer)
                                buffer.clear()
                                chunk = convert_to_html(chunk)
                                yield f"data: {chunk}\n\n"
                if buffer:
                    chunk = ''.join(buffer)
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
        conversational_rag_chain = RunnableWithMessageHistory(
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
    print("history",history)
    chat_history = [
        {"role": "AI" if isinstance(msg, AIMessage) else "Human", "content": msg.content}
        for msg in history.messages
    ]
    return jsonify({"chat_history": chat_history})


if __name__ == '__main__':
    
    app.run(host="0.0.0.0",debug=False,port=50002)