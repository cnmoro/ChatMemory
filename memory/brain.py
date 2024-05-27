from memory.embeddings import AlternativeModel, \
    extract_embeddings_free, extract_embeddings_openai
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

        Args with defaults:
        - mongita_storage_location: The location where the Mongita database will be stored.
        - vector_db_storage_location: The location where the vector database will be stored.
        - embedding_extraction_function: A function that extracts embeddings from text.
        - free_embedding_model_type: The type of the free embedding model to use.
        - summarization_function: A function that summarizes text.
        - openai_key: The OpenAI API key.
        - openai_embedding_model: The OpenAI embedding model to use.
        - openai_summarization_model: The OpenAI summarization model to use.
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

    def get_last_interactions(self, session_id, num_chats=4, recent_first=True):
        """
        Retrieves the last N chats from the database.
        """
        chats = self.db_coll.find(
            {"session_id": session_id}
        ).sort('timestamp', -1 if recent_first else 1).limit(num_chats)
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
        # Retrieve the N most recent pairs of questions and answers
        last_n_messages = self.get_last_interactions(session_id, recent_interaction_count)
        last_n_messages_ids = [ m['message_id'] for m in last_n_messages ]

        # Get embeddings for the incoming prompt
        prompt_embedding = self.extract_embeddings_wrapper(new_prompt)

        # Search in vector database for the most similar question
        # (Excluding the last "N" messages, as they are fetched directly from the database)
        _, _, metadatas = self.vector_db.find_most_similar(
            prompt_embedding,
            metadata_filter={'session_id': session_id},
            k = 10
        )
        metadatas = [ m for m in metadatas if m['message_id'] not in last_n_messages_ids ][:2]

        suggested_context = ""
        if len(metadatas) > 0:
            for metadata in metadatas:
                suggested_context += f"Previous context ({'prompt' if metadata['type'] == 'question' else 'answer'}): {metadata['sentence']}\n"

        suggested_context += "\n"

        if len(last_n_messages) > 0:
            last_n_messages.reverse()
            for message in last_n_messages:
                if 'question' in message:
                    suggested_context += f"Previous prompt: {message['question_summary']}\n"
                else:
                    suggested_context += f"Previous answer: {message['answer_summary']}\n"
        
        # Return the context metadata
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
        # Delete from the database
        self.db_coll.delete_many({"session_id": session_id})

        # Delete from the vector database
        self.delete_session_from_vector_db(session_id)
    
    def forget_message(self, session_id, message_id):
        """
        Forgets the given message.
        """
        # Delete from the database
        self.db_coll.delete_one({"session_id": session_id, "message_id": message_id})

        # Delete from the vector database
        self.delete_message_from_vector_db(session_id, message_id)

    def list_messages(self, session_id, count = False, page = 1, limit = 20, recent_first = True):
        """
        Lists the messages for the given session.
        Order is from most recent to oldest.
        """
        skip = (page - 1) * limit
        messages = self.db_coll.find(
            {"session_id": session_id}
        ).sort('timestamp', -1 if recent_first else 1).skip(skip).limit(limit)
        messages = list(messages)

        if count:
            return self.db_coll.count_documents({"session_id": session_id})
        else:
            return messages
