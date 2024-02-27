[![codecov](https://codecov.io/gh/cnmoro/ChatMemory/graph/badge.svg?token=B75N22ACYP)](https://codecov.io/gh/cnmoro/ChatMemory)

[![Downloads](https://static.pepy.tech/badge/chatmemorydb)](https://pepy.tech/project/chatmemorydb)

[![Downloads](https://static.pepy.tech/badge/chatmemorydb/month)](https://pepy.tech/project/chatmemorydb)

[![Downloads](https://static.pepy.tech/badge/chatmemorydb/week)](https://pepy.tech/project/chatmemorydb)

## **Memory**

This is a Python project aimed at providing an extremely simple way to manage the “memory” of a chatbot, when using LLMs. The goal is to provide a simple interface to memorize prompts and answers, as well as store it using **summarization**, while providing methods to fetch the context of the conversation - considering both **short** and **long-term** memory. Please check out the code snippets below.

Memory is stored in two forms, a simple vector database for semantic search, as well as Mongita (which is a "sqlite" for MongoDB) for chat history and other metadata.

### **Installation**

```plaintext
pip install chatmemorydb
```

### **Usage**

```python
from memory.brain import Memory

memory = Memory()

# Start a new session
session_id, question_id, answer_id = memory.memorize("Hello", "Hi there! How can I help you?")

# Memorize new interactions with the new session id
memory.memorize("What is the capital of Italy?", "The capital of Italy is Rome.", session_id)
memory.memorize("What is the capital of France?", "The capital of France is Paris.", session_id)
memory.memorize("What is the capital of Spain?", "The capital of Spain is Madrid.", session_id)
memory.memorize("What is the capital of Brazil?", "The capital of Brazil is Brasília.", session_id)
memory.memorize("What is the capital of Peru?", "The capital of Peru is Lima.", session_id)
memory.memorize("What is the capital of Colombia?", "The capital of Colombia is Bogotá.", session_id)
memory.memorize("What is the capital of Venezuela?", "The capital of Venezuela is Caracas.", session_id)
_, question_id, answer_id = memory.memorize("What is the capital of Ecuador?", "The capital of Ecuador is Quito.", session_id)

# Forget a given message
memory.forget_message(session_id, answer_id)

# List existing messages
messages = memory.list_messages(session_id, page=1, limit=3, recent_first=True)

# Result
[
    {
        '_id': ObjectId('65d789c5f698b7e4198e4b1f'),
        'message_id': '77942ef3-3a63-4e3c-90d3-f4c13ffef1ec',
        'question': 'What is the capital of Ecuador?',
        'question_summary': 'What is the capital of Ecuador?',
        'session_id': 'c8ea1875-9e77-4eb7-8f2a-a1dfc7c104c5',
        'timestamp': datetime.datetime(2024, 2, 22, 17, 52, 5, 930335)
    },
    {
        '_id': ObjectId('65d789c5f698b7e4198e4b1e'),
        'answer': 'The capital of Venezuela is Caracas.',
        'answer_summary': 'The capital of Venezuela is Caracas.',
        'message_id': 'bacd8e4b-0d1d-477d-bd23-f99f12068224',
        'session_id': 'c8ea1875-9e77-4eb7-8f2a-a1dfc7c104c5',
        'timestamp': datetime.datetime(2024, 2, 22, 17, 52, 5, 906237)
    },
    {
        '_id': ObjectId('65d789c5f698b7e4198e4b1d'),
        'message_id': 'a6256250-3797-4073-ae85-3d803fbc8f27',
        'question': 'What is the capital of Venezuela?',
        'question_summary': 'What is the capital of Venezuela?',
        'session_id': 'c8ea1875-9e77-4eb7-8f2a-a1dfc7c104c5',
        'timestamp': datetime.datetime(2024, 2, 22, 17, 52, 5, 905285)
    }
]

# Remember something (core functionality)
new_prompt = "I love Madrid! What can you tell me about it?"
retrieved_memory = memory.remember(session_id, new_prompt)

# Result
{
    # We get the most recent interactions by default (short-term memory)
    'recent_memory': [
        {
            'session_id': 'c8ea1875-9e77-4eb7-8f2a-a1dfc7c104c5',
            'message_id': '77942ef3-3a63-4e3c-90d3-f4c13ffef1ec',
            'question': 'What is the capital of Ecuador?',
            'question_summary': 'What is the capital of Ecuador?',
            'timestamp': datetime.datetime(2024, 2, 22, 17, 52, 5, 930335)
        },
        {
            'session_id': 'c8ea1875-9e77-4eb7-8f2a-a1dfc7c104c5',
            'message_id': 'bacd8e4b-0d1d-477d-bd23-f99f12068224',
            'answer': 'The capital of Venezuela is Caracas.',
            'answer_summary': 'The capital of Venezuela is Caracas.',
            'timestamp': datetime.datetime(2024, 2, 22, 17, 52, 5, 906237)
        },
        ...
    ],
    # We also retrieve the most similar questions and answers from the long-term memory
    # using semantic search
    'context_memory': [
        {
            'sentence': 'the capital of spain is madrid.',
            'session_id': 'c8ea1875-9e77-4eb7-8f2a-a1dfc7c104c5',
            'message_id': 'f39886b8-f5d8-4610-a4a5-fd833572bfae',
            'type': 'answer'
        },
        {
            'sentence': 'what is the capital of spain ?',
            'session_id': 'c8ea1875-9e77-4eb7-8f2a-a1dfc7c104c5',
            'message_id': '96644ee7-c388-49f5-89b8-6e03905ef92c',
            'type': 'question'
        }
    ],
    
    # Then, everything is glued together and provided as a suggestion
    # This is the text that would be injected into the new prompt,
    # as context of the existing conversation
    'suggested_context': '''
        Previous context (answer): the capital of spain is madrid.
        Previous context (prompt): what is the capital of spain ?
        
        Previous prompt: What is the capital of Ecuador?
        Previous answer: The capital of Venezuela is Caracas.
        Previous prompt: What is the capital of Venezuela?
        Previous answer: The capital of Colombia is Bogotá.
    '''
}

# Forget an entire session
memory.forget_session(session_id)

### Parameters for the memory object

### Customize the storage locations
# param mongita_storage_location : defaults to './mongita_memory' (folder)
# param vector_db_storage_location : defaults to './vector_memory.pkl' (file)

### Custom embedding function (bring your own!)
### Should be able to receive the text and return a list of embeddings
### If no OpenAI key is provided, a free model will be used
# param free_embedding_model_type: AlternativeModel = AlternativeModel.tiny
# param embedding_extraction_function = None

### Custom summarization (bring your own!)
### Should be able to receive the text and return the summary
# param summarization_function = None

### OpenAI key and models
### Default ones are the most cost-effective
# param openai_key: str = None
# param openai_embedding_model: str = "text-embedding-3-small"
# param openai_summarization_model: str = "gpt-3.5-turbo"
```

## **License**

This project is licensed under the MIT License.
