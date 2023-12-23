import pickle 
from pathlib import Path
import faiss
from .pre_processor import PreProcessor

class LoadIndex():
    def __init__(self, year):
        self.file_name = f"{year}.pdf"
        self.pre_processor = PreProcessor(file_year=year)
        self.paragraphs = self.pre_processor.chunk_text()

    def load_index(self):
        pickle_path = Path.cwd() / Path(f"prophet/pickles/{self.file_name}.pkl")
        loaded = None
        try:
            with open(pickle_path, "rb") as f:
                loaded = faiss.deserialize_index(pickle.load(f))
        except:
            loaded = self.pre_processor.embed()
        return (loaded, self.pre_processor, self.paragraphs)
        