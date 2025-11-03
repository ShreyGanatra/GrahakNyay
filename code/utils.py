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
import re


# get current path
current_path = os.path.dirname(os.path.realpath(__file__))

# NEW ADDITION - KIRAN
# ======================== START ==========================

import docx
from langchain.docstore.document import Document

def load_sectoral_docs():
    """Load and chunk all SectoralQA/*.docx files."""
    sector_path = os.path.join(current_path, 'SectoralQA')
    sector_docs = {}
    print("\nLoading sectoral docs...")
    for fname in os.listdir(sector_path):
        if fname.endswith('.docx'):
            sector = fname.replace('.docx', '')
            doc_path = os.path.join(sector_path, fname)
            text = extract_docx_text(doc_path)
            docs = [Document(page_content=chunk, metadata={"source": sector}) for chunk in text.split("\n\n") if chunk.strip()]
            sector_docs[sector] = docs
    print("\nsector_docs: ", sector_docs.keys())
    # print(f"\n[Sample] Sector Docs for {sector}:\n{sector_docs[sector]}")
    return sector_docs

# def extract_docx_text(path):
#     doc = docx.Document(path)
#     return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])

def extract_docx_text(path):
    doc = docx.Document(path)
    rels = doc.part.rels
    output = []

    for para in doc.paragraphs:
        para_text = ""
        for run in para.runs:
            # Check if the run is part of a hyperlink
            r_element = run._element
            hlink_click = r_element.xpath(".//w:hyperlink")
            if hlink_click:
                for hlink in hlink_click:
                    r_id = hlink.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                    if r_id and r_id in rels:
                        link = rels[r_id].target_ref
                        link_text = run.text.strip()
                        if link_text:
                            para_text += f"{link_text} ({link})"
            else:
                para_text += run.text
        if para_text.strip():
            output.append(para_text.strip())

    return "\n".join(output)



def load_few_shot_examples():
    """For each sector, load two chats and associated legal docs as text."""
    root = os.path.join(current_path, 'SectoralClassified')
    sectors = os.listdir(os.path.join(root, 'Chat'))
    sector_examples = {}
    for sector in sectors:
        try:
            chat_files = os.listdir(os.path.join(root, 'Chat', sector))
            examples = []
            for fname in chat_files[:5]:
                base = fname.replace('.docx', '')
                chat = extract_docx_text(os.path.join(root, 'Chat', sector, fname))
                complaint = extract_docx_text(os.path.join(root, 'Complaint', sector, fname))
                notice = extract_docx_text(os.path.join(root, 'Notice', sector, fname))
                # affidavit = extract_docx_text(os.path.join(root, 'Affidavits', sector, fname))
                # mop = extract_docx_text(os.path.join(root, 'MoP', sector, fname))
                # examples.append(f"Chat:\n{chat}\n\nLegal Notice:\n{notice}\n\nLegal Complaint:\n{complaint}\n\nAffidavit:\n{affidavit}\n\nMoP:\n{mop}")
                examples.append(f"Chat:\n{chat}\n\nNotice:\n{notice}\n\nComplaint:\n{complaint}\n")
            sector_examples[sector] = "\n\n".join(examples)
        except Exception as e:
            print(f"Error loading sector {sector}: {e}")
            continue
    return sector_examples

# ========================= END ===========================




def get_chunks(qa_path):
    path_to_csv = os.path.join(current_path, qa_path)
    loader = CSVLoader(file_path=path_to_csv, 
        metadata_columns=["File"],
        source_column="File",
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
        # model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda:0"},
    )
    vectorstore = FAISS.from_documents(documents=doc_chunks, embedding=embeddings)
    return vectorstore

# ========================= START ===========================


# def get_llm(model):

#     # tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
#     # model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
#     # streamer = TextStreamer(tok,skip_prompt=True)

#     # pipe = pipeline(
#     #     task="text-generation",
#     #     model = model,
#     #     tokenizer = tok,
#     #     # streamer = streamer,
#     #     temperature = 0.0,
#     #     pad_token_id=tok.eos_token_id,
#     #     device=0,
#     #     return_full_text=False,
#     #     max_new_tokens=2048,
#     #     do_sample=False,
#     # )

#     # llm = HuggingFacePipeline(
#     #     pipeline = pipe,
#     #     model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     # )

#     # llm_engine_hf = ChatHuggingFace(llm=llm, tokenizer=tok)
#     # llm = HuggingFaceEndpoint(
#     #     endpoint_url="localhost:8080/v1/chat/completions",
#     #     streaming=True,
#     # )

#     # llm_engine_hf = ChatHuggingFace(llm=llm)
#     llm_engine_hf = ChatOpenAI(
#         # model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
#         openai_api_key="EMPTY",
#         openai_api_base="http://172.17.0.1:8080/v1/",
#         temperature=0,
#         # max_tokens=2048,
#         )
#     return llm_engine_hf

from langchain_groq import ChatGroq
# pip install langchain-groq
from langchain_google_genai import ChatGoogleGenerativeAI
# pip install langchain-google-genai
from dotenv import load_dotenv
# Load .env file
load_dotenv()

def get_llm(model: str=None):
    """
    Return an LLM instance based on the given model name.
    - "hosted" -> locally hosted model
    - anything else -> Groq API model
    """

    # import os
    # from openai import OpenAI

    # client = OpenAI(
    #     base_url="https://router.huggingface.co/v1",
    #     api_key=os.environ["HF_TOKEN"],
    # )

    # completion = client.chat.completions.create(
    #     model="openai/gpt-oss-20b:fireworks-ai",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "What is the capital of France?"
    #         }
    #     ],
    # )

    # print(completion.choices[0].message)
    if not model or model.lower() == "hosted":
        # Locally hosted LLM
        return ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://172.17.0.1:8080/v1/",
            temperature=0,
        )
    elif "hf-model-" in model:
        return ChatOpenAI(
            model = model.replace('hf-model-', ''),
            openai_api_key=os.getenv("HF_TOKEN"),
            openai_api_base="https://router.huggingface.co/v1",
            temperature=0,
        )

    elif "gemini" in model:
        # Google Gemini LLM
        print(f"Using Google Gemini model: {model}")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key,
            temperature=0,
            # convert_system_message_to_human=True, # Helps with compatibility for some chat prompts
        )
    elif 'groq' in model:
        # Groq LLM
        model = model.replace('groq-', '')
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        return ChatGroq(
            model=model,
            groq_api_key=groq_api_key,
            temperature=0,
        )
    
# ========================= END ===========================


def get_conversation_chain(retriever, llm_engine_hf, one_shot=False, general_corpus=False, rag=True, sector_prompt=None):

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
    system_prompt = sector_prompt if sector_prompt else get_prompt(one_shot=one_shot, general_corpus=general_corpus)

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
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    text = text.replace('\n', '<br>')
    return text
    
    # # Handle bold, italic, and code with proper opening tags
    # text = text.replace('**', '<b>', 1)
    # text = text.replace('*', '<i>', 1)
    # text = text.replace('`', '<code>', 1)

    replacements = {
        '\n': '<br>',
        '**': '</b>',  # Bold
        '*': '</i>',   # Italic
        '`': '</code>' # Code
    }
    
    # Apply all other replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Make links clickable
    # link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    # text = link_pattern.sub(r'<a href="\2">\1</a>', text)
    
    # Make plain text links clickable
    # plain_link_pattern = re.compile(r'(http[s]?://\S+)')
    # text = plain_link_pattern.sub(r'<a href="\1">\1</a>', text)

    return text
