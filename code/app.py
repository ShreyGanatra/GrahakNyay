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
from utils import get_chunks, get_vectorstore, get_llm, get_conversation_chain
import uuid


BASE_URL = '/consumer_chatbot'
vectorstore = None
consumer_store = {}
consumer_conversation_chain = None

def get_consumer_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in consumer_store:
        consumer_store[session_id] = ChatMessageHistory()
        initial_message = "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
        consumer_store[session_id].add_message(AIMessage(content=initial_message))
    return consumer_store[session_id]


def initialize_app():
    global vectorstore, consumer_conversation_chain
    print("Initializing application...")
    text_chunks = get_chunks()
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever(
        # search_type="mmr",
        # search_kwargs={'k': 10, 'fetch_k': 50}
    )
    print("Documents processed successfully")
    llm_engine_hf = get_llm()
    consumer_conversation_chain = get_conversation_chain(retriever,llm_engine_hf)
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
    
    history = get_consumer_session_history(session_id)
    initial_ai_message = history.messages[0].content if history.messages else "Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you."
    return jsonify({"message": initial_ai_message})


@app.route(f'{BASE_URL}/chat', methods=['POST', 'GET'])
def consumer_chat():
    return chat_handler(consumer_conversation_chain, get_consumer_session_history, rag=True)


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
        response = chain.invoke({"input":user_input},config)
        formatted_response = response['answer'].replace('\n', '<br>')

        return jsonify({"response": formatted_response}), 200
    
@app.route(f'{BASE_URL}/get_chat_history', methods=['GET'])
def get_consumer_chat_history():
    return get_chat_history(get_consumer_session_history)


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
    
    app.run(host="0.0.0.0",debug=False,port=50012)