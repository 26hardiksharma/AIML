from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self,model_name = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        print(f"Embedding Dimensions: {self.model.get_embedding_dimension()}")

    def generate_embedding(self,text):
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings