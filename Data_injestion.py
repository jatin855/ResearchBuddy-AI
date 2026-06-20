from src import load_document,create_chunks,Embedding_Manager,Vector_Store

doc = load_document(director_path="PDFs")
chunks = create_chunks(doc)

embedding_manager = Embedding_Manager()

Documents = []
for i in chunks:
    Documents.append(i.page_content)

embeddings = embedding_manager.gen_embeddings(Documents)

vector_dB = Vector_Store(persistent_Directory="vector_dB")
vector_dB.add_document(chunks,embeddings)


