from sentence_transformers import SentenceTransformer
import faiss
from pdfminer.high_level import extract_text
from pathlib import Path
import os
import openai
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
import argparse

def extract(fname):
    current_directory = Path.cwd()
    input_path = current_directory / Path(f"prophet/source/{fname}")
    return extract_text(input_path)

def chunk_text(text_input, max_token_size):
    text_input = [s.strip() for s in text_input.splitlines() if s.strip()]
    tokenizer = tiktoken.encoding_for_model('gpt-4')

    # Encode the text_data into token integers
    token_integers = tokenizer.encode(" ".join(text_input))

    # Split the token integers into chunks based on max_token_size
    chunks = [
        token_integers[i : i + max_token_size]
        for i in range(0, len(token_integers), max_token_size)
    ]

    # Decode token chunks back to strings
    return [tokenizer.decode(chunk) for chunk in chunks]
    
def embed(fname):
    text = extract(fname)
    paragraphs = chunk_text(text, max_token_size=1000)
    
    encoder_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    encoder_model.max_seq_length = 512

    embeddings = encoder_model.encode(
        paragraphs,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    #cosine similarity search for performance
    faiss_index = faiss.IndexFlatIP(encoder_model.get_sentence_embedding_dimension())
    faiss_index.add(embeddings)
    return encoder_model, faiss_index, paragraphs


def generate_answer(encoder_model, faiss_index, paragraphs, question="What number is represented here by 168 Billion?"):
    base_prompt = f"""You are a financial analyst. You are going to be looking at the Microsoft 10k for 2022. A client is asking questions about Microsoft's performance. Please answer to the best of your ability, only AFTER seeing the 'ALL_PARTS_SENT' indicator.

    Your answers are correct, high-quality, and written by an domain expert. 

    User question: {{}}

    Contexts:
    {{}}
    """
    k = 6
    decoder_model_name="gpt-3.5-turbo"
    query_embedding = encoder_model.encode([question])
    distances, neighbors = faiss_index.search(query_embedding, k)

    # provide the top k nearest neighbors as context
    relevant_context = [f"{i}. {paragraphs[index]}" for i, index in enumerate(neighbors[0])]
    responses = []
    messages = [
        {"role": "system", "content": base_prompt},
        {
            "role": "user",
            "content": f"QUESTION: {question}. Do not give an answer yet.",
        },
    ]
    tokenizer = tiktoken.encoding_for_model('gpt-4')
    decoder_model = OpenAI()

    # Append least accurate to most accurate, popping least accurate if we hit token length
    for i in range(len(relevant_context)-1, -1, -1):
        chunk = relevant_context[i]
        messages.append({"role": "user", "content": f"context: {chunk}. Do not give an answer yet."})

        # Check if total tokens exceed the model's limit and remove least relevant chunk. Do not remove system context or question.
        while (sum(len(tokenizer.encode(msg["content"])) for msg in messages) > 4090):
            messages.pop(2)

        response = decoder_model.chat.completions.create(model=decoder_model_name, messages=messages)
        chatgpt_response = response.choices[0].message.content.strip()
        responses.append(chatgpt_response)

    messages.append({"role": "user", "content": "context: {context}. ALL PARTS SENT. Please analze the entire context and give the answer."})
    response = decoder_model.chat.completions.create(model=decoder_model_name, messages=messages)
    final_response = response.choices[0].message.content.strip()
    responses.append(final_response)
    print(responses[-1])
    print("----------")
    return responses[-1]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, required=True)
    return parser.parse_args()

def start():
    fiscal_year = parse_args().year
    if fiscal_year not in ["2022", "2023"]:
        raise "please restart this program with the --year=XXXX modifier. Currently year may be 2022 or 2023"
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model, faiss_index, paragraphs = embed(f"{fiscal_year}.pdf")

    is_valid_question = True
    print(f"Welcome to earnings review! This chatbot will allow you to comb through the pdf of Microsoft's annual financial statements for the year {fiscal_year}")
    print("Please ask your questions and I will do my best to answer.")
    
        
    while is_valid_question:
        question= input("Please ask your question\n\n")
        if question == "exit" or question == "quit":
            is_valid_question = False
        elif "?" not in question:
            print("Please try asking your question again with '?' at the end of the sentence.")
        else:
            generate_answer(model, faiss_index, paragraphs, question)
    
