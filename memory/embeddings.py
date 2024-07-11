from minivectordb.embedding_model import EmbeddingModel

model = EmbeddingModel(onnx_model_cpu_core_count=1)

def extract_embeddings(text):
    return model.extract_embeddings(text)
