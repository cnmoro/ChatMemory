from memory.embeddings import AlternativeModel, extract_embeddings_free, model

text = "Hello, world!"

def test_extract_embeddings_free_tiny():
    result = extract_embeddings_free(text, AlternativeModel.tiny, reload_model=True)
    assert len(result) == 512

def test_extract_embeddings_free_small():
    result = extract_embeddings_free(text, AlternativeModel.small, reload_model=True)
    assert len(result) == 384