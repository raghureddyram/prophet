from sentence_transformers import SentenceTransformer
import faiss
from pdfminer.high_level import extract_text
from pathlib import Path
import tiktoken

class PreProcessor():
    def __init__(self, file_year):
        self.file_name = Path.cwd() / Path(f"prophet/source/{file_year}.pdf")
        self.file_text = extract_text(self.file_name)
        self.max_token_size = 1000

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

    def embed(self):
        paragraphs = self.chunk_text()
        
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
        return encoder_model, faiss_index