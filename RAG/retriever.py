class RAGRetriever:
    def __init__(self,embedding_manager,vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self,query,top_k = 5,score_threshold=0.0):
        query_embeddings = self.embedding_manager.generate_embedding([query])[0]
        results = self.vector_store.collection.query(
             query_embeddings = [query_embeddings.tolist()],
             n_results = top_k
        )

        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents= results["documents"][0]
            distances= results["distances"][0]

            for i,(id,metadata,document,distance) in enumerate(zip(ids,metadatas,documents,distances)):

                similarity = 1 - distance

                if(similarity>=score_threshold):
                    retrieved_docs.append({
                        "id":id,
                        "document":document,
                        "metadata":metadata,
                        "distance":distance,
                        "similarity":similarity,
                        "rank":i+1
                    })
                print(f"Distance: {distance:.4f} | Similarity: {similarity:.4f}")
            print(f"Retrieved {len(retrieved_docs)} documents.")

        else:
            print("No documents found...")

        return retrieved_docs

