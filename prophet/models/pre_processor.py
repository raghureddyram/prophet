from sentence_transformers import SentenceTransformer
import faiss
from pdfminer.high_level import extract_text
from pathlib import Path
import tiktoken
import pickle

class PreProcessor():
    def __init__(self, file_year, encoder_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")):
        self.file_name = f"{file_year}.pdf"
        self.file_path = Path.cwd() / Path(f"prophet/source/{file_year}.pdf")
        self.file_text = extract_text(self.file_path)
        self.max_token_size = 1000
        self.encoder_model = encoder_model

    def chunk_text(self):
        text_input = [s.strip() for s in self.file_text.splitlines() if s.strip()]
        tokenizer = tiktoken.encoding_for_model('gpt-4')

        # Encode the text_data into token integers
        token_integers = tokenizer.encode(" ".join(text_input))

        # Split the token integers into chunks based on max_token_size
        chunks = [
            token_integers[i : i + self.max_token_size]
            for i in range(0, len(token_integers), self.max_token_size)
        ]

        # Decode token chunks back to strings
        return [tokenizer.decode(chunk) for chunk in chunks]
    
    def save_pickle(self, obj):
        pickle_path = Path.cwd() / Path(f"prophet/pickles/{self.file_name}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(obj, f)

    def embed(self):
        paragraphs = self.chunk_text()
        
        self.encoder_model.max_seq_length = 512

        embeddings = self.encoder_model.encode(
            paragraphs,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        #cosine similarity search for performance

        faiss_index = faiss.IndexFlatIP(self.encoder_model.get_sentence_embedding_dimension())
        faiss_index.add(embeddings)

        pickleable = faiss.serialize_index(faiss_index)
        self.save_pickle(pickleable)
        return faiss_index