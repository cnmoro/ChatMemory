from memory.embeddings import AlternativeModel, extract_embeddings_free, \
    extract_embeddings_openai
from memory.summarization import summarize_text_basic, \
    summarize_text_with_gpt
from minivectordb.vector_database import VectorDatabase
from mongita import MongitaClientDisk
from datetime import datetime
from openai import OpenAI
from text_util_en_pt.cleaner import structurize_text
import uuid

class Memory:
    def __init__(
            self,
            mongita_storage_location: str = './mongita_memory',
            vector_db_storage_location: str = './vector_memory.pkl',
            embedding_extraction_function = None,
            free_embedding_model_type: AlternativeModel = AlternativeModel.tiny,
            summarization_function = None,
            openai_key: str = None,
            openai_embedding_model: str = "text-embedding-3-small",
            openai_summarization_model: str = "gpt-3.5-turbo"):
        """
        Initializes a new instance of the class.

        Args:
            db_name (str, optional): The name of the database. Defaults to "chat_db".
            embedding_extraction_function (function, optional): The function to use for extracting embeddings from text.
            free_embedding_model_type (AlternativeModel, optional): The type of the free embedding model to use. Defaults to AlternativeModel.tiny.
            summarization_function (function, optional): The function to use for summarizing text.
            openai_key (str, optional): The API key for OpenAI. If provided, an OpenAI client will be initialized.
            openai_embedding_model (str, optional): The name of the OpenAI embedding model to use. Defaults to "text-embedding-3-small".
            openai_summarization_model (str, optional): The name of the OpenAI summarization model to use. Defaults to "gpt-3.5-turbo".
        """
        self.embedding_extraction_function = embedding_extraction_function
        self.summarization_function = summarization_function
        self.openai_key = openai_key
        self.free_embedding_model_type = free_embedding_model_type
        self.openai_embedding_model = openai_embedding_model
        self.openai_summarization_model = openai_summarization_model
        self.vector_db_storage_location = vector_db_storage_location
        self.vector_db = VectorDatabase(storage_file=vector_db_storage_location)
        self.db_location = mongita_storage_location

        if self.openai_key is not None:
            self.openai_client = OpenAI(api_key=self.openai_key)
        
        if self.embedding_extraction_function is None and self.openai_key is None:
            # Generate a sample embedding to force the loading of the model
            self.DUMMY_EMBEDDING = extract_embeddings_free("Hello", self.free_embedding_model_type)
        else:
            self.DUMMY_EMBEDDING = self.extract_embeddings_wrapper("Hello")

        self.db_client = MongitaClientDisk(self.db_location)
        self.db_coll = self.db_client["chat_db"]["chat_sessions"]

        # Ensure index creation
        self.db_coll.create_index([("session_id", 1)])
        self.db_coll.create_index([("timestamp", 1)])

    def summarize(self, text):
        """
        Summarizes the given text.
        """
        if self.summarization_function is not None:
            return self.summarization_function(text)
        else:
            if self.openai_key is not None:
                return summarize_text_with_gpt(text, self.openai_client, self.openai_summarization_model)
            else:
                return summarize_text_basic(text)

    def store_embeddings(self, sentences, session_id, message_id, type):
        for sentence in sentences:
            sentence_embedding = self.extract_embeddings_wrapper(sentence)
            self.vector_db.store_embedding(
                str(uuid.uuid4()),
                sentence_embedding,
                {
                    'sentence': sentence,
                    'session_id': session_id,
                    'message_id': message_id,
                    'type': type
                }
            )

    def memorize(self, question, answer, session_id=None):
        """
        Adds a chat interaction to the database.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())  # Generate a new session ID for the first chat message pair

        question_id = str(uuid.uuid4())
        question_summary = self.summarize(question)

        answer_id = str(uuid.uuid4())
        answer_summary = self.summarize(answer)

        self.db_coll.insert_one({
            "session_id": session_id,
            "message_id": question_id,
            "question": question,
            "question_summary": question_summary,
            "timestamp": datetime.utcnow()
        })

        self.db_coll.insert_one({
            "session_id": session_id,
            "message_id": answer_id,
            "answer": answer,
            "answer_summary": answer_summary,
            "timestamp": datetime.utcnow()
        })

        # Add the pair to the vector database
        question_sentences = [ t['sentence'] for t in structurize_text(question_summary, sentence_length=200)]
        answer_sentences = [ t['sentence'] for t in structurize_text(answer_summary, sentence_length=200)]

        self.store_embeddings(question_sentences, session_id, question_id, 'question')
        self.store_embeddings(answer_sentences, session_id, answer_id, 'answer')

        return session_id, question_id, answer_id

    def get_last_interactions(self, session_id, num_chats=4):
        """
        Retrieves the last N chats from the database.
        """
        chats = self.db_coll.find(
            {"session_id": session_id}
        ).sort('timestamp', -1).limit(num_chats)
        chats = list(chats)

        # Remove the _id field
        # (Mongita does not support projection, so we have to do this manually)
        for chat in chats:
            chat.pop('_id')

        return chats
        
    def extract_embeddings_wrapper(self, text):
        if self.embedding_extraction_function is not None:
            prompt_embedding = self.embedding_extraction_function(text)
        elif self.openai_key is not None:
            prompt_embedding =  extract_embeddings_openai(text, self.openai_client, self.openai_embedding_model)
        else:
            prompt_embedding = extract_embeddings_free(text, self.free_embedding_model_type)

        return prompt_embedding

    def remember(self, session_id, new_prompt, recent_interaction_count = 4):
        """
        Fetches relevant information from the database based on the new prompt.
        """
        # 1 - Retrieve the N most recent pairs of questions and answers
        last_n_messages = self.get_last_interactions(session_id, recent_interaction_count)
        last_n_messages_ids = [ m['message_id'] for m in last_n_messages ]

        # 2 - Get embeddings for the incoming prompt
        prompt_embedding = self.extract_embeddings_wrapper(new_prompt)

        # 3 - Search in vector database for the most similar question
        # (Excluding the last "N" messages, as they are fetched directly from the database)
        _, _, metadatas = self.vector_db.find_most_similar(
            prompt_embedding,
            k = 10
        )
        metadatas = [ m for m in metadatas if m['message_id'] not in last_n_messages_ids ][:2]

        suggested_context = ""
        if len(metadatas) > 0:
            for metadata in metadatas:
                suggested_context += f"Previous context ({'prompt' if metadata['type'] == 'question' else 'answer'}): {metadata['sentence']}\n"

        suggested_context += "\n"

        if len(last_n_messages) > 0:
            for message in last_n_messages:
                if 'question' in message:
                    suggested_context += f"Previous prompt: {message['question_summary']}\n"
                else:
                    suggested_context += f"Previous answer: {message['answer_summary']}\n"
        
        # 4 - Return the context metadata
        return {
            "recent_memory": last_n_messages,
            "context_memory": metadatas,
            "suggested_context": suggested_context.strip()
        }

    def delete_session_from_vector_db(self, session_id):
        ids, _, _ = self.vector_db.find_most_similar(
            self.DUMMY_EMBEDDING,
            metadata_filter={'session_id': session_id},
            k=9999
        )

        # Remove all ids
        [ self.vector_db.delete_embedding(id) for id in ids ]

    def delete_message_from_vector_db(self, session_id, message_id):
        ids, _, _ = self.vector_db.find_most_similar(
            self.DUMMY_EMBEDDING,
            metadata_filter={'session_id': session_id, 'message_id': message_id},
            k=2
        )

        # Remove all ids
        [ self.vector_db.delete_embedding(id) for id in ids ]

    def forget_session(self, session_id):
        """
        Forgets the given session.
        """
        # 1 - Delete from the database
        self.db_coll.delete_many({"session_id": session_id})

        # 2 - Delete from the vector database
        self.delete_session_from_vector_db(session_id)
    
    def forget_message(self, session_id, message_id):
        """
        Forgets the given message.
        """
        # 1 - Delete from the database
        self.db_coll.delete_one({"session_id": session_id, "message_id": message_id})

        # 2 - Delete from the vector database
        self.delete_message_from_vector_db(session_id, message_id)

# if __name__ == '__main__':
#     # Example usage
#     memory = Memory()

#     session_id, question_id, answer_id = memory.memorize("Hello. My name is Carlo!", "Hi there Carlo, what can I do for you?")

#     # Memorize more interactions
#     memory.memorize("What is the capital of France?", "The capital of France is Paris.", session_id)
#     memory.memorize("What is the capital of Spain?", "The capital of Spain is Madrid.", session_id)
#     memory.memorize("What is the capital of Portugal?", "The capital of Portugal is Lisbon.", session_id)
#     # More
#     memory.memorize("What do you like to do on friday?", "I like to go to the movies.", session_id)
#     memory.memorize("What is your favorite movie?", "My favorite movie is The Matrix.", session_id)
#     memory.memorize("What is your favorite food?", "I like pizza.", session_id)

#     # Remember
#     retrieved_memory = memory.remember(session_id, "What is the capital of Italy?")
