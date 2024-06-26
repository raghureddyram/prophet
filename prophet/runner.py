import os
import openai
from dotenv import load_dotenv
import argparse
from .models import AnswerGenerator, LoadIndex

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, required=True)
    return parser.parse_args()

def initialize_chatbot(fiscal_year: str) -> AnswerGenerator:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    faiss_index, pre_processor, paragraphs  = LoadIndex(fiscal_year).load_index()
    encoder_model = pre_processor.encoder_model
    return AnswerGenerator(encoder_model=encoder_model, index=faiss_index, paragraphs=paragraphs)

def start() -> None:
    fiscal_year = parse_args().year
    if fiscal_year not in ["2022", "2023"]:
        raise ValueError("Please restart this program with the --year=XXXX modifier. Currently year may be 2022 or 2023")

    answer_generator = initialize_chatbot(fiscal_year)

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
            answer_generator.generate_answer(question)

