from contextlib import contextmanager
from memory.brain import Memory
from memory.embeddings import AlternativeModel
from minivectordb.vector_database import VectorDatabase
from memory.summarization import summarize_text_basic
import numpy as np
import shutil, os

OPENAI_KEY = os.environ.get('OPENAI_KEY')

@contextmanager
def get_memory_object(**kwargs):
    memory = Memory(**kwargs)
    yield memory
    memory.db_client.close()

    # Remove the created files and folders
    if os.path.exists(memory.db_location):
        shutil.rmtree(memory.db_location)

    if os.path.exists(memory.vector_db_storage_location):
        os.remove(memory.vector_db_storage_location)

def test_memory_initialization():
    with get_memory_object() as memory:
        assert memory.openai_key is None
        assert memory.free_embedding_model_type == AlternativeModel.tiny
        assert isinstance(memory.vector_db, VectorDatabase)
        assert memory.db_location == './mongita_memory'

# Test for summarization functionality
def test_summarize():
    with get_memory_object() as memory:
        summary = memory.summarize("This is a test sentence for summarization.")
        assert isinstance(summary, str)

# Test for memorize and retrieval of interactions
def test_memorize_and_retrieve():
    with get_memory_object() as memory:
        session_id, question_id, answer_id = memory.memorize("Test question", "Test answer")
        assert isinstance(session_id, str)
        
        interactions = memory.get_last_interactions(session_id, num_chats=2)
        print(interactions)
        assert isinstance(interactions, list)
        assert len(interactions) == 2

        # Answer comes first, because it is the most recent
        assert interactions[0]['answer'] == "Test answer"
        assert interactions[1]['question'] == "Test question"

# Test for embedding extraction wrapper
def test_extract_embeddings_wrapper():
    with get_memory_object() as memory:
        embedding = memory.extract_embeddings_wrapper("This is a test sentence.")
        assert isinstance(embedding, list) or isinstance(embedding, np.ndarray)

# Test forgetting a session
def test_forget_session():
    with get_memory_object() as memory:
        session_id, question_id, answer_id = memory.memorize("Question for forgetting", "Answer for forgetting")
        memory.forget_session(session_id)
        interactions_after_forgetting = memory.get_last_interactions(session_id, num_chats=1)
        assert len(interactions_after_forgetting) == 0

# Test remembering based on new prompt
def test_remember():
    with get_memory_object() as memory:
        session_id, question_id, answer_id = memory.memorize("Hello. My name is X!", "Hi there X, what can I do for you?")
        retrieved_memory = memory.remember(session_id, "What is the capital of Italy?")
        assert isinstance(retrieved_memory, dict)
        assert 'recent_memory' in retrieved_memory
        assert 'context_memory' in retrieved_memory
        assert 'suggested_context' in retrieved_memory
        assert isinstance(retrieved_memory['recent_memory'], list)
        assert isinstance(retrieved_memory['context_memory'], list)
        assert isinstance(retrieved_memory['suggested_context'], str)

def test_summarize_with_provided_function():
    with get_memory_object(summarization_function=summarize_text_basic) as memory:
        summary = memory.summarize("This is a test sentence for summarization.")
        assert isinstance(summary, str)

def test_extract_embedding_with_provided_function():
    with get_memory_object(embedding_extraction_function=lambda x: [1, 2, 3]) as memory:
        embedding = memory.extract_embeddings_wrapper("This is a test sentence.")
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        assert embedding == [1, 2, 3]

def test_remember_without_memory():
    with get_memory_object() as memory:
        retrieved_memory = memory.remember("123", "This is a test sentence.")
        assert retrieved_memory['recent_memory'] == []
        assert retrieved_memory['context_memory'] == []
        assert retrieved_memory['suggested_context'] == ''

def test_delete_something_from_memory():
    with get_memory_object() as memory:
        session_id, question_id, answer_id = memory.memorize("My name is X", "Hello X! My name is Chatbot")

        memory.forget_message(session_id, answer_id)
        interactions = memory.get_last_interactions(session_id, num_chats=1)
        assert len(interactions) == 1

        # Remember and assert that the answer is not in the context memory
        retrieved_memory = memory.remember(session_id, "Name")
        assert 'Chatbot' not in retrieved_memory['suggested_context']

        # Now delete the question as well
        memory.forget_message(session_id, question_id)
        interactions = memory.get_last_interactions(session_id, num_chats=1)
        assert len(interactions) == 0

def test_remember_bigger_conversation():
    with get_memory_object() as memory:
        session_id, _, _ = memory.memorize("Hello", "Hi there! How can I help you?")
        memory.memorize("What is the capital of Italy?", "The capital of Italy is Rome.", session_id)
        memory.memorize("What is the capital of France?", "The capital of France is Paris.", session_id)
        memory.memorize("What is the capital of Spain?", "The capital of Spain is Madrid.", session_id)
        memory.memorize("What is the capital of Germany?", "The capital of Germany is Berlin.", session_id)
        memory.memorize("What is the capital of the United Kingdom?", "The capital of the United Kingdom is London.", session_id)
        memory.memorize("What is the capital of the United States?", "The capital of the United States is Washington D.C.", session_id)
        memory.memorize("What is the capital of Canada?", "The capital of Canada is Ottawa.", session_id)
        memory.memorize("What is the capital of Mexico?", "The capital of Mexico is Mexico City.", session_id)
        memory.memorize("What is the capital of Brazil?", "The capital of Brazil is Brasília.", session_id)
        memory.memorize("What is the capital of Argentina?", "The capital of Argentina is Buenos Aires.", session_id)
        memory.memorize("What is the capital of Chile?", "The capital of Chile is Santiago.", session_id)
        memory.memorize("What is the capital of Peru?", "The capital of Peru is Lima.")
        memory.memorize("What is the capital of Colombia?", "The capital of Colombia is Bogotá.", session_id)
        memory.memorize("What is the capital of Venezuela?", "The capital of Venezuela is Caracas.", session_id)
        memory.memorize("What is the capital of Ecuador?", "The capital of Ecuador is Quito.", session_id)

        # Now remember the conversation
        retrieved_memory = memory.remember(session_id, "What is the capital of Brazil ?")

        print(retrieved_memory)

        assert 'brasília' in retrieved_memory['suggested_context'].lower()

def test_brain_summarization_with_openai():
    with get_memory_object(openai_key=OPENAI_KEY) as memory:
        summary = memory.summarize("This is a test sentence for summarization.")
        assert isinstance(summary, str)

def test_brain_embedding_extraction_with_openai():
    with get_memory_object(openai_key=OPENAI_KEY) as memory:
        embedding = memory.extract_embeddings_wrapper("This is a test sentence.")
        assert isinstance(embedding, list) or isinstance(embedding, np.ndarray)

def test_list_and_count_messages_with_pagination():
    with get_memory_object() as memory:
        # Memorize a few messages
        session_id, _, _ = memory.memorize("Hello", "Hi there! How can I help you?")
        memory.memorize("What is the capital of Italy?", "The capital of Italy is Rome.", session_id)
        memory.memorize("What is the capital of France?", "The capital of France is Paris.", session_id)
        memory.memorize("What is the capital of Spain?", "The capital of Spain is Madrid.", session_id)
        memory.memorize("What is the capital of Germany?", "The capital of Germany is Berlin.", session_id)

        # Test count
        count = memory.list_messages(session_id, count=True)
        assert count == 10

        # Test list
        messages = memory.list_messages(session_id, page=1, limit=4)
        assert len(messages) == 4

def test_multiple_sessions_ensure_no_interference():
    with get_memory_object() as memory:
        # Memorize a few messages
        session_1, _, _ = memory.memorize("Hello, my name is John Doe", "Hi there John! How can I help you?")
        memory.memorize("What is the capital of Italy?", "The capital of Italy is Rome.", session_1)
        memory.memorize("What is the capital of France?", "The capital of France is Paris.", session_1)
        memory.memorize("What is the capital of Spain?", "The capital of Spain is Madrid.", session_1)

        # Now start a new session
        session_2, _, _ = memory.memorize("Hello, my name is Jane Doe", "Hi there Jane! How can I help you?")
        memory.memorize("What is the capital of Germany?", "The capital of Germany is Berlin.", session_2)
        memory.memorize("What is the capital of the United Kingdom?", "The capital of the United Kingdom is London.", session_2)
        memory.memorize("What is the capital of the United States?", "The capital of the United States is Washington D.C.", session_2)

        # Now remember the conversation for session 1
        retrieved_memory = memory.remember(session_1, "capital of Italy ?")
        # Assert that no information from session_2 is present
        assert 'berlin' not in retrieved_memory['suggested_context'].lower()
        assert 'london' not in retrieved_memory['suggested_context'].lower()
        assert 'washington' not in retrieved_memory['suggested_context'].lower()
        assert 'germany' not in retrieved_memory['suggested_context'].lower()
        assert 'united kingdom' not in retrieved_memory['suggested_context'].lower()
        assert 'united states' not in retrieved_memory['suggested_context'].lower()

        # Now remember the conversation for session 2
        retrieved_memory = memory.remember(session_2, "capital of the United States ?")
        # Assert that no information from session_1 is present
        assert 'rome' not in retrieved_memory['suggested_context'].lower()
        assert 'paris' not in retrieved_memory['suggested_context'].lower()
        assert 'madrid' not in retrieved_memory['suggested_context'].lower()
        assert 'italy' not in retrieved_memory['suggested_context'].lower()
        assert 'france' not in retrieved_memory['suggested_context'].lower()
        assert 'spain' not in retrieved_memory['suggested_context'].lower()
        