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
        'session_id': '1c4c9102-2d01-47d6-9204-1038a7f19ea7',
    'message_id': 'fdbec0e0-ef92-4f6d-9e05-f3881a7b6c4b',
    'question': 'What is the capital of Ecuador?',
    'question_summary': 'What is the capital of Ecuador?',
    'answer': None,
    'answer_summary': None,
    'timestamp': '2024-07-11 16:58:01.004009'
    },
    {
        'session_id': '1c4c9102-2d01-47d6-9204-1038a7f19ea7',
        'message_id': '5ce5c6e3-db3c-4a72-aba3-94f5500b2e27',
        'question': None,
        'question_summary': None,
        'answer': 'The capital of Venezuela is Caracas.',
        'answer_summary': 'The capital of Venezuela is Caracas.',
        'timestamp': '2024-07-11 16:58:00.993597'
    },
    {
        'session_id': '1c4c9102-2d01-47d6-9204-1038a7f19ea7',
        'message_id': '1897cbab-7e6f-4c85-b3fe-bd60a1f02172',
        'question': 'What is the capital of Venezuela?',
        'question_summary': 'What is the capital of Venezuela?',
        'answer': None,
        'answer_summary': None,
        'timestamp': '2024-07-11 16:58:00.993499'
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
            'session_id': '1c4c9102-2d01-47d6-9204-1038a7f19ea7',
            'message_id': '5fd79a44-c110-43b7-b4ef-2ab59a1c03fa',
            'question': None,
            'question_summary': None,
            'answer': 'The capital of Colombia is Bogotá.',
            'answer_summary': 'The capital of Colombia is Bogotá.',
            'timestamp': '2024-07-11 16:58:00.983392'
        },
        {
            'session_id': '1c4c9102-2d01-47d6-9204-1038a7f19ea7',
            'message_id': '1897cbab-7e6f-4c85-b3fe-bd60a1f02172',
            'question': 'What is the capital of Venezuela?',
            'question_summary': 'What is the capital of Venezuela?',
            'answer': None,
            'answer_summary': None,
            'timestamp': '2024-07-11 16:58:00.993499'
        },
        {
            'session_id': '1c4c9102-2d01-47d6-9204-1038a7f19ea7',
            'message_id': '5ce5c6e3-db3c-4a72-aba3-94f5500b2e27',
            'question': None,
            'question_summary': None,
            'answer': 'The capital of Venezuela is Caracas.',
            'answer_summary': 'The capital of Venezuela is Caracas.',
            'timestamp': '2024-07-11 16:58:00.993597'
        },
        {
            'session_id': '1c4c9102-2d01-47d6-9204-1038a7f19ea7',
            'message_id': 'fdbec0e0-ef92-4f6d-9e05-f3881a7b6c4b',
            'question': 'What is the capital of Ecuador?',
            'question_summary': 'What is the capital of Ecuador?',
            'answer': None,
            'answer_summary': None,
            'timestamp': '2024-07-11 16:58:01.004009'
        }
    ],
    # We also retrieve the most similar questions and answers from the long-term memory
    # using semantic search
    'context_memory': [
        {
            'sentence': 'The capital of Spain is Madrid.',
            'session_id': '1c4c9102-2d01-47d6-9204-1038a7f19ea7',
            'message_id': '39cf7746-3324-4348-916a-06e45e161867',
            'type': 'answer'
        },
        {
            'sentence': 'What is the capital of Spain?',
            'session_id': '1c4c9102-2d01-47d6-9204-1038a7f19ea7',
            'message_id': '5f88719c-d037-445a-8423-71c1482ce016',
            'type': 'question'
        }
    ],
    # Then, everything is glued together and provided as a suggestion
    # This is the text that would be injected into the new prompt,
    # as context of the existing conversation
    'suggested_context': '''
    Previous context (answer): The capital of Spain is Madrid.
    Previous context (prompt): What is the capital of Spain?
    
    Previous answer: The capital of Colombia is Bogotá.
    Previous prompt: What is the capital of Venezuela?
    Previous answer: The capital of Venezuela is Caracas.
    Previous prompt: What is the capital of Ecuador?
    '''
}


# Forget an entire session
memory.forget_session(session_id)
