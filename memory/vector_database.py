import numpy as np, pymongo, traceback, faiss, multiprocessing

class VectorDB:
    def __init__(self, mongo_uri: str, mongo_database: str, mongo_collection: str, vector_storage, text_storage):
        self.mongo_uri = mongo_uri
        self.mongo_database = mongo_database
        self.mongo_collection = mongo_collection
        self.mongo_reference = pymongo.MongoClient(self.mongo_uri)[self.mongo_database][self.mongo_collection]
        self.vector_storage = vector_storage
        self.text_storage = text_storage

    def check_counts(self):
        print(f"Semantic storage count: {self.vector_storage.get_data_count()}")
        print(f"Text storage count: {self.text_storage.get_data_count()}")
        print(f"MongoDB count: {self.mongo_reference.count_documents({})}")
    
    def get_total_count(self):
        return self.text_storage.get_data_count()

    def ensure_embeddings_typing(self, embeddings):
        # Ensure embeddings is a numpy array
        if type(embeddings) is not np.ndarray:
            # Convert embeddings to numpy array 32-bit float
            embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings

    def store_embeddings_batch(self, unique_ids: list, embeddings, metadata_dicts=[], text_field=None):
        payload = []

        embeddings = self.ensure_embeddings_typing(embeddings)

        if len(metadata_dicts) < len(unique_ids):
            metadata_dicts.extend([{} for _ in range(len(unique_ids) - len(metadata_dicts))])
        
        if text_field is not None:
            texts = [ m.pop(text_field, '') for m in metadata_dicts ]
        else:
            texts = [ '' for _ in range(len(unique_ids)) ]

        for uid, metadata_dict in zip(unique_ids, metadata_dicts):
            payload.append({**{ '_id': uid }, **metadata_dict})

        self.vector_storage.store_vectors(embeddings, unique_ids)
        self.text_storage.store_data(texts, unique_ids)
        self.mongo_reference.insert_many(payload)
    
    def delete_embeddings_batch(self, unique_ids):
        self.mongo_reference.delete_many({'_id': {'$in': unique_ids}})
        self.vector_storage.delete_data(unique_ids)
        self.text_storage.delete_data(unique_ids)
    
    def delete_embeddings_by_metadata(self, metadata_filters):
        identifiers = list(self.mongo_reference.find(metadata_filters, {'_id': 1}))
        self.mongo_reference.delete_many(metadata_filters)
        self.vector_storage.delete_data([i['_id'] for i in identifiers])
        self.text_storage.delete_data([i['_id'] for i in identifiers])
    
    def get_vector_by_metadata(self, metadata_filters):
        try:
            first_result = self.mongo_reference.find_one({**metadata_filters})
            if first_result is None:
                return None
            return self.vector_storage.get_vectors([first_result['_id']])[0]
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        return None

    def search_faiss(sef, query_embeddings, corpus_embeddings, top_k):
        faiss.normalize_L2(corpus_embeddings)
        
        index = faiss.IndexHNSWFlat(corpus_embeddings.shape[1], 16)

        # Optimize graph construction speed
        index.hnsw.efConstruction = 50 # Lower value for faster construction (default is 200)

        # Use multi-threading for faster processing
        faiss.omp_set_num_threads(multiprocessing.cpu_count() // 2)

        index.add(corpus_embeddings)

        distances, indices = index.search(query_embeddings, top_k)

        results = []
        
        # Zip the indices and distances together
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            results.append({
                "corpus_id": idx,
                "score": dist
            })
        
        return results

    def find_most_similar(self, embedding, filters={}, output_fields=[], k=5, use_find_one=False):
        """
        Main entry point to find the most similar documents based on the given embedding.
        """
        try:
            # Step 1: Get documents from MongoDB
            results = self._fetch_mongo_documents(filters, output_fields, use_find_one)
            if not results:
                return [], [], []

            # Step 2: Prepare embeddings
            vector_ids, vectors, query_embedding, = self._prepare_embeddings(results, embedding)

            if not vector_ids:
                return [], [], []
            
            # Create a vector_id to index mapping
            vector_id_to_index = { idx: i for idx, i in enumerate(vector_ids) }
            # Step 3: Perform similarity search
            semantic_results = self.search_faiss(
                query_embeddings = query_embedding, 
                corpus_embeddings = vectors,
                top_k = k
            )
            ids = [ r['corpus_id'] for r in semantic_results if r['corpus_id'] != -1 ]
            scores =  [ r['score'] for r in semantic_results if r['corpus_id'] != -1 ]
            translated_ids = [ vector_id_to_index[i] for i in ids ]

            db_ids = translated_ids
            scores = scores

            # Iterate db_ids and scores together. If a duplicate db_id is found, remove it and its score
            seen_ids = set()
            
            # Iterate from inverse order to prevent errors on removal during iteration
            for i in range(len(db_ids) - 1, -1, -1):
                if db_ids[i] in seen_ids:
                    db_ids.pop(i)
                    scores.pop(i)
                else:
                    seen_ids.add(db_ids[i])

            # Step 4: Prepare final results
            return self._prepare_final_results(db_ids, scores, results)

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            return [], [], []

    def _fetch_mongo_documents(self, filters, output_fields, use_find_one, limit=None, skip=None):
        """Handle MongoDB document retrieval"""

        if output_fields == 'all':
            projection = {}
        else:
            output_fields = list(set(output_fields)) + ['_id', 'index']
            projection = {field: 1 for field in output_fields}

        if use_find_one:
            doc = self.mongo_reference.find_one(filters, projection)
            return [doc] if doc else []
        
        cursor = self.mongo_reference.find(filters, projection)
        if limit is not None:
            cursor = cursor.limit(limit)
        if skip is not None:
            cursor = cursor.skip(skip)
        result = list(cursor)
        return result

    def _prepare_embeddings(self, results, query_embedding):
        """Prepare embeddings for similarity search"""
        vector_ids = [r['_id'] for r in results]

        retrieved_vectors = self.vector_storage.get_vectors(vector_ids)
        # Some vectors could be None, zip together with vector_ids and remove the indices on both lists where the value is None
        zipped_vectors = zip(vector_ids, retrieved_vectors)
        filtered_vectors = [v for v in zipped_vectors if v[1] is not None]
        if not filtered_vectors:
            return [], [], []
        vector_ids, retrieved_vectors = zip(*filtered_vectors)

        lmdb_vectors = np.array(retrieved_vectors, dtype=np.float32)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        return vector_ids, lmdb_vectors, query_embedding

    def _prepare_final_results(self, db_ids, scores, mongo_results):
        """Prepare the final results with texts and metadata"""
        texts = self.text_storage.get_data(db_ids)
        id_mapped_results = {r['_id']: r for r in mongo_results}
        
        metadatas = [
            id_mapped_results[db_id] if db_id in id_mapped_results else {}
            for db_id in db_ids
        ]

        for metadata, text in zip(metadatas, texts):
            metadata['text'] = text

        return db_ids, list(scores), metadatas
