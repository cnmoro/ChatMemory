from memory.embeddings import AlternativeModel, extract_embeddings_free

text = "Hello, world!"

def test_extract_embeddings_free_tiny():
    result = extract_embeddings_free(text, AlternativeModel.tiny)
    assert len(result) == 512

def test_extract_embeddings_free_small():
    result = extract_embeddings_free(text, AlternativeModel.small)
    assert len(result) == 512