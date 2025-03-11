from memory.lmdb_vector_mapping import LmdbStorage, MemmapStorage
from memory.embeddings import get_onnx_embeddings
import uuid, pymongo, time, datetime, signal
from memory.vector_database import VectorDB

class Memory:
    def __init__(
            self,
            mongo_uri: str,
            mongo_database: str,
            mongo_collection_vectordb: str,
            mongo_collection_conversation_data: str,
            override_vector_storage = None,
            override_text_storage = None
        ):
        
        if override_vector_storage is None and override_text_storage is None:
            # Create instances of the MemmapStorage class
            override_vector_storage = MemmapStorage("mem_vector_storage")
            override_text_storage = LmdbStorage("mem_text_storage")
        
        def signal_handler(signum, frame):
            print("Received signal, closing storage")
            override_vector_storage.close()
            override_text_storage.close()
            print("Closed storage")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.vectordb = VectorDB(
            mongo_uri = mongo_uri,
            mongo_database = mongo_database,
            mongo_collection = mongo_collection_vectordb,
            vector_storage = override_vector_storage,
            text_storage = override_text_storage
        )
        self.semantic_vectors_storage = override_vector_storage
        self.semantic_texts_storage = override_text_storage
        self.mongo_database = mongo_database
        self.mongo_collection_conversation_data = mongo_collection_conversation_data
        self.mongo_collection_vectordb = mongo_collection_vectordb
        self.connection = pymongo.MongoClient(mongo_uri)
        self.conversation_collection = self.connection[mongo_database][mongo_collection_conversation_data]
            
    def store_embeddings(self, sentences: list, session_id: str, message_id: str, type: str):
        unique_ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
        embeddings = get_onnx_embeddings(sentences)
        metadatas = [
            {
                'text': sentence,
                'session_id': session_id,
                'message_id': message_id,
                'type': type
            }
            for sentence in sentences
        ]

        self.vectordb.store_embeddings_batch(
            unique_ids = unique_ids,
            embeddings = embeddings,
            metadata_dicts = metadatas,
            text_field = 'text'
        )

    def memorize(self, question, answer, session_id=None):
        if session_id is None:
            session_id = str(uuid.uuid4())

        question_id = str(uuid.uuid4())

        answer_id = str(uuid.uuid4())

        self.conversation_collection.insert_one({
            'session_id': session_id,
            'message_id': question_id,
            'question': question,
            'timestamp': datetime.datetime.now(datetime.timezone.utc)
        })

        # Sleep for 10ms so the answer is inserted after the question
        time.sleep(0.01)

        self.conversation_collection.insert_one({
            'session_id': session_id,
            'message_id': answer_id,
            'answer': answer,
            'timestamp': datetime.datetime.now(datetime.timezone.utc)
        })

        self.store_embeddings([question], session_id, question_id, 'question')
        self.store_embeddings([answer], session_id, answer_id, 'answer')

        return session_id, question_id, answer_id

    def get_last_interactions(self, session_id, num_chats=4, recent_first=True):
        chats = list(self.conversation_collection.find(
            {'session_id': session_id},
            sort = [('timestamp', -1 if recent_first else 1)],
            limit = num_chats
        ))

        # Convert to dictionary format
        columns = ['session_id', 'message_id', 'question', 'answer', 'timestamp']

        # Ensure all items contains all columns, if not, fill with None
        for chat in chats:
            for column in columns:
                if column not in chat:
                    chat[column] = None

        return chats

    def remember(self, session_id, new_prompt, recent_interaction_count = 4):
        """
        Fetches relevant information from the database based on the new prompt.
        """
        # Retrieve the N most recent pairs of questions and answers
        last_n_messages = self.get_last_interactions(session_id, recent_interaction_count)
        last_n_messages_ids = [ m['message_id'] for m in last_n_messages ]

        # Get embeddings for the incoming prompt
        prompt_embedding = get_onnx_embeddings([new_prompt])[0]

        # Search in vector database for the most similar question
        # (Excluding the last "N" messages, as they are fetched directly from the database)
        _, _, metadatas = self.vectordb.find_most_similar(
            embedding = prompt_embedding,
            filters = {'session_id': session_id},
            output_fields = 'all',
            k = 30,
            use_find_one = False
        )
        if not metadatas:
            return {
                "recent_memory": [],
                "context_memory": [],
                "suggested_context": ""
            }
        
        metadatas = [ m for m in metadatas if m['message_id'] not in last_n_messages_ids ][:2]

        suggested_context = ""
        if len(metadatas) > 0:
            for metadata in metadatas:
                suggested_context += f"Previous context ({'prompt' if metadata['type'] == 'question' else 'answer'}): {metadata['text']}\n"

        suggested_context += "\n"

        if len(last_n_messages) > 0:
            last_n_messages.reverse()
            for message in last_n_messages:
                if 'question' in message and bool(message['question']):
                    suggested_context += f"Previous prompt: {message['question']}\n"
                else:
                    suggested_context += f"Previous answer: {message['answer']}\n"
        
        # Return the context metadata
        return {
            "recent_memory": last_n_messages,
            "context_memory": metadatas,
            "suggested_context": suggested_context.strip()
        }

    def delete_session_from_vector_db(self, session_id):
        self.vectordb.delete_embeddings_by_metadata({'session_id': session_id})

    def delete_message_from_vector_db(self, session_id, message_id):
        self.vectordb.delete_embeddings_by_metadata({'session_id': session_id, 'message_id': message_id})

    def forget_session(self, session_id):
        self.conversation_collection.delete_many({'session_id': session_id})

        # Delete from the vector database
        self.delete_session_from_vector_db(session_id)
    
    def forget_message(self, session_id, message_id):
        self.conversation_collection.delete_one({'session_id': session_id, 'message_id': message_id})

        # Delete from the vector database
        self.delete_message_from_vector_db(session_id, message_id)

    def list_messages(self, session_id, count = False, page = 1, limit = 20, recent_first = True):
        if count:
            return self.conversation_collection.count_documents({'session_id': session_id})
        else:
            offset = (page - 1) * limit
            order = -1 if recent_first else 1
            messages = list(self.conversation_collection.find(
                {'session_id': session_id},
                sort = [('timestamp', order)],
                limit = limit,
                skip = offset
            ))

            return messages
