from minivectordb.sharded_vector_database import ShardedVectorDatabase
from memory.compression import compress_text, structurize_text
from memory.embeddings import extract_embeddings
import uuid, sqlite3, numpy as np, threading
from datetime import datetime

dummy_embedding = np.zeros(512, dtype=np.float32)

class Memory:
    def __init__(
            self,
            sqlite_db_path: str = './memory.db',
            vector_db_storage_folder_location: str = 'memory_shards'
        ):
        """
        Initializes a new instance of the class.

        Args with defaults:
        - sqlite_db_path: The path to the SQLite database file.
        - vector_db_storage_location: The location where the vector database will be stored.
        """
        self.vector_db_storage_folder_location = vector_db_storage_folder_location
        self.vector_db = ShardedVectorDatabase(storage_dir=vector_db_storage_folder_location)
        self.sqlite_db_path = sqlite_db_path
        self.lock = threading.Lock()

        self.init_db()

    def init_db(self):
        with self.lock:
            with sqlite3.connect(self.sqlite_db_path) as db_conn:
                cursor = db_conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT,
                        message_id TEXT PRIMARY KEY,
                        question TEXT,
                        question_summary TEXT,
                        answer TEXT,
                        answer_summary TEXT,
                        timestamp DATETIME
                    )
                ''')
                db_conn.commit()

                # Create index for faster lookups
                cursor = db_conn.cursor()
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS chat_sessions_session_id_index ON chat_sessions (session_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS chat_sessions_message_id_index ON chat_sessions (message_id)
                ''')
                db_conn.commit()
            
    def store_embeddings(self, sentences, session_id, message_id, type):
        unique_ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
        embeddings = [extract_embeddings(sentence) for sentence in sentences]
        metadatas = [
            {
                'sentence': sentence,
                'session_id': session_id,
                'message_id': message_id,
                'type': type
            }
            for sentence in sentences
        ]

        self.vector_db.store_embeddings_batch(unique_ids, embeddings, metadatas)

    def memorize(self, question, answer, session_id=None):
        if session_id is None:
            session_id = str(uuid.uuid4())

        question_id = str(uuid.uuid4())
        question_summary = compress_text(question)

        answer_id = str(uuid.uuid4())
        answer_summary = compress_text(answer)

        with self.lock:
            with sqlite3.connect(self.sqlite_db_path) as db_conn:
                cursor = db_conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_sessions (session_id, message_id, question, question_summary, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, question_id, question, question_summary, datetime.utcnow()))

                cursor.execute('''
                    INSERT INTO chat_sessions (session_id, message_id, answer, answer_summary, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, answer_id, answer, answer_summary, datetime.utcnow()))

                db_conn.commit()

        # Add the pair to the vector database
        question_sentences = structurize_text(question_summary)
        answer_sentences = structurize_text(answer_summary)

        self.store_embeddings(question_sentences, session_id, question_id, 'question')
        self.store_embeddings(answer_sentences, session_id, answer_id, 'answer')

        return session_id, question_id, answer_id

    def get_last_interactions(self, session_id, num_chats=4, recent_first=True):
        with self.lock:
            with sqlite3.connect(self.sqlite_db_path) as db_conn:
                cursor = db_conn.cursor()
                order = 'DESC' if recent_first else 'ASC'
                cursor.execute(f'''
                    SELECT * FROM chat_sessions
                    WHERE session_id = ?
                    ORDER BY timestamp {order}
                    LIMIT ?
                ''', (session_id, num_chats))
                chats = cursor.fetchall()

        # Convert to dictionary format
        columns = ['session_id', 'message_id', 'question', 'question_summary', 'answer', 'answer_summary', 'timestamp']
        return [dict(zip(columns, chat)) for chat in chats]

    def remember(self, session_id, new_prompt, recent_interaction_count = 4):
        """
        Fetches relevant information from the database based on the new prompt.
        """
        # Retrieve the N most recent pairs of questions and answers
        last_n_messages = self.get_last_interactions(session_id, recent_interaction_count)
        last_n_messages_ids = [ m['message_id'] for m in last_n_messages ]

        # Get embeddings for the incoming prompt
        prompt_embedding = extract_embeddings(new_prompt)

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
                if 'question' in message and bool(message['question']):
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
            dummy_embedding,
            metadata_filter={'session_id': session_id},
            k=9999
        )

        self.vector_db.delete_embeddings_batch(list(ids))

    def delete_message_from_vector_db(self, session_id, message_id):
        ids, _, _ = self.vector_db.find_most_similar(
            dummy_embedding,
            metadata_filter={'session_id': session_id, 'message_id': message_id},
            k=2
        )

        # Remove all ids
        self.vector_db.delete_embeddings_batch(list(ids))

    def forget_session(self, session_id):
        with self.lock:
            with sqlite3.connect(self.sqlite_db_path) as db_conn:
                cursor = db_conn.cursor()
                cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
                db_conn.commit()

        # Delete from the vector database
        self.delete_session_from_vector_db(session_id)
    
    def forget_message(self, session_id, message_id):
        with self.lock:
            with sqlite3.connect(self.sqlite_db_path) as db_conn:
                cursor = db_conn.cursor()
                cursor.execute('DELETE FROM chat_sessions WHERE session_id = ? AND message_id = ?', (session_id, message_id))
                db_conn.commit()

        # Delete from the vector database
        self.delete_message_from_vector_db(session_id, message_id)

    def list_messages(self, session_id, count = False, page = 1, limit = 20, recent_first = True):
        with self.lock:
            with sqlite3.connect(self.sqlite_db_path) as db_conn:
                cursor = db_conn.cursor()
        
                if count:
                    cursor.execute('SELECT COUNT(*) FROM chat_sessions WHERE session_id = ?', (session_id,))
                    return cursor.fetchone()[0]
                else:
                    offset = (page - 1) * limit
                    order = 'DESC' if recent_first else 'ASC'
                    cursor.execute(f'''
                        SELECT * FROM chat_sessions
                        WHERE session_id = ?
                        ORDER BY timestamp {order}
                        LIMIT ? OFFSET ?
                    ''', (session_id, limit, offset))
                    messages = cursor.fetchall()

                    # Convert to dictionary format
                    columns = ['session_id', 'message_id', 'question', 'question_summary', 'answer', 'answer_summary', 'timestamp']
                    return [dict(zip(columns, message)) for message in messages]
