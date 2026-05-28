import chromadb
import uuid
import os


class VectorStoreManager:
    def __init__(self,persist_dir = "data/vector_store",collection_name = "pdf_documents"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.collection = None
        self.client = None

        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_dir,exist_ok=True)

        self.client = chromadb.PersistentClient(path = self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            metadata={
                "description":"Vector Store for PDF Embeddings",
                "hnsw:space":"cosine"
            }
        )

        print(f"Vector Store Initialized @{self.collection_name} || Docs in collection: {self.collection.count()}")

    def add_documents(self,documents,embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents doesn't match number of embeddings!")
        
        # Store ids, embeddings, document, metadata
        ids = []
        all_metadata = []
        documents_content = []
        embeddings_list = []

        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["context_length"] = len(doc.page_content)
            all_metadata.append(metadata)

            documents_content.append(doc.page_content)

            embeddings_list.append(embedding.tolist())
            
        self.collection.add(
            ids = ids,
            metadatas = all_metadata,
            documents=documents_content,
            embeddings=embeddings_list
        )

        print(f"Total documents added: {len(documents_content)} || Total count: {self.collection.count()}")
        