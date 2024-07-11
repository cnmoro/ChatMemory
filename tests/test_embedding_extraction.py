from memory.embeddings import extract_embeddings

text = "Hello, world!"

def test_extract_embeddings():
    result = extract_embeddings(text)
    assert len(result) == 512
