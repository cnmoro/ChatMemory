import msgpack, struct, lmdb, time, os, atexit
import numpy as np

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken for {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper

class LmdbStorage:
    def __init__(self, path, map_size=70*1024*1024*1024): # 70GB by default
        self.env = lmdb.open(path, map_size=map_size)
        atexit.register(self.close)

    def _int_to_bytes(self, x):
        return struct.pack('<q', x)

    def store_data(self, data, identifiers, batch_size=5000):
        total = len(data)
        with self.env.begin(write=True) as txn:
            for i in range(0, total, batch_size):
                batch_data = data[i:i+batch_size]
                batch_ids = identifiers[i:i+batch_size]
                with txn.cursor() as curs:
                    for vec, id in zip(batch_data, batch_ids):
                        curs.put(
                            self._int_to_bytes(id) if isinstance(id, int) else id.encode(),
                            msgpack.packb(vec)
                        )

    def get_data(self, identifiers):
        datas = []
        with self.env.begin() as txn:
            for id in identifiers:
                data = txn.get(self._int_to_bytes(id) if isinstance(id, int) else id.encode())
                if data:
                    datas.append(msgpack.unpackb(data))
                else:
                    datas.append(None)
        return [ v for v in datas if v is not None ]
    
    def delete_data(self, identifiers):
        with self.env.begin(write=True) as txn:
            for id in identifiers:
                txn.delete(self._int_to_bytes(id) if isinstance(id, int) else id.encode())

    def get_data_count(self):
        with self.env.begin() as txn:
            return txn.stat()['entries']
        
    def sync(self):
        self.env.sync()

    def close(self):
        self.env.close()

class MemmapStorage:
    def __init__(self, path, dtype=np.float32):
        self.env = lmdb.open(path, map_size=70*1024*1024*1024)
        self.memmap_path = path + "_data.dat"
        self.dtype = dtype
        self.itemsize = np.dtype(self.dtype).itemsize
        self._initialize_memmap()

    def _initialize_memmap(self):
        # Ensure memmap file exists and has at least one item
        if not os.path.exists(self.memmap_path):
            # Initialize memmap file with a small size (e.g., one item)
            initial_size = 1
            with open(self.memmap_path, 'wb') as f:
                f.write(b'\x00' * self.itemsize * initial_size)
            self.current_size = initial_size
        else:
            # Get current size from memmap file
            filesize = os.path.getsize(self.memmap_path)
            self.current_size = filesize // self.itemsize
            if self.current_size == 0:
                # Same as above, ensure there's at least one item to mmap
                initial_size = 1
                with open(self.memmap_path, 'wb') as f:
                    f.write(b'\x00' * self.itemsize * initial_size)
                self.current_size = initial_size

        # Open memmap with the current size
        self.memmap_array = np.memmap(
            self.memmap_path, dtype=self.dtype, mode='r+', shape=(self.current_size,))

    def _int_to_bytes(self, x):
        return struct.pack('<q', x)

    def _merge_free_regions(self, free_list):
        # Sort the free list by offset
        free_list = sorted(free_list, key=lambda x: x[0])
        merged_free_list = []
        for region in free_list:
            if not merged_free_list:
                merged_free_list.append(region)
            else:
                last_offset, last_size = merged_free_list[-1]
                current_offset, current_size = region
                if last_offset + last_size == current_offset:
                    # Regions are adjacent, merge them
                    merged_free_list[-1] = (last_offset, last_size + current_size)
                else:
                    merged_free_list.append(region)
        return merged_free_list

    def store_vectors(self, vectors, identifiers):
        vectors = [np.asarray(v, dtype=self.dtype).ravel() for v in vectors]
        sizes = [v.size for v in vectors]
        # Fetch the free list from LMDB
        with self.env.begin(write=True) as txn:
            free_list_data = txn.get(b'free_list')
            if free_list_data:
                free_list = msgpack.unpackb(free_list_data, use_list=False, strict_map_key=False)
                free_list = [tuple(free_region) for free_region in free_list]
            else:
                free_list = []

            allocations = []
            for size in sizes:
                # Try to find a free region that can fit the vector
                for i, (free_offset, free_size) in enumerate(free_list):
                    if free_size >= size:
                        # Use this free region
                        allocations.append((free_offset, size))
                        # Update the free region
                        if free_size > size:
                            # Shrink the free region
                            free_list[i] = (free_offset + size, free_size - size)
                        else:
                            # Remove the free region
                            free_list.pop(i)
                        break
                else:
                    # No suitable free region, allocate at the end
                    allocations.append((self.current_size, size))
                    self.current_size += size

            # Resize memmap file if current_size has increased
            new_file_size = self.current_size * self.itemsize
            with open(self.memmap_path, 'rb+') as f:
                f.truncate(new_file_size)

            # Re-open the memmap with the new size
            self.memmap_array = np.memmap(
                self.memmap_path, dtype=self.dtype, mode='r+', shape=(self.current_size,))

            # Write vectors and record offsets
            for (vec, id, (offset, size)) in zip(vectors, identifiers, allocations):
                # Write vector to memmap
                self.memmap_array[offset:offset+size] = vec
                # Store offset and size in LMDB
                key = self._int_to_bytes(id) if isinstance(id, int) else id.encode()
                value = struct.pack('<QQ', offset, size)
                txn.put(key, value)

            # Save the updated free list
            free_list_packed = msgpack.packb(free_list)
            txn.put(b'free_list', free_list_packed)

    def get_vectors(self, identifiers):
        vectors = [None] * len(identifiers)
        # Map to store the original indices of the keys
        key_indices = {}
        keys = []

        for idx, id in enumerate(identifiers):
            key = self._int_to_bytes(id) if isinstance(id, int) else id.encode()
            keys.append(key)
            key_indices[key] = idx

        # Sort the keys since getmulti works efficiently with ordered keys
        sorted_keys = sorted(keys)

        with self.env.begin(buffers=True) as txn:
            with txn.cursor() as cursor:
                results = cursor.getmulti(sorted_keys)
                for key, value in results:
                    if key in key_indices:
                        idx = key_indices[key]
                        if value:
                            offset, size = struct.unpack('<QQ', value)
                            vec = self.memmap_array[offset:offset + size]
                            vectors[idx] = vec.copy()  # Copy to return a normal ndarray
                        else:
                            vectors[idx] = None
                        # Remove the key from the keys to search
                        del key_indices[key]
                        # Break if we've found all keys
                        if not key_indices:
                            break
        return vectors

    def delete_data(self, identifiers):
        with self.env.begin(write=True) as txn:
            # Load the free list
            free_list_data = txn.get(b'free_list')
            if free_list_data:
                free_list = msgpack.unpackb(free_list_data, use_list=False, strict_map_key=False)
                free_list = [tuple(free_region) for free_region in free_list]
            else:
                free_list = []

            for id in identifiers:
                key = self._int_to_bytes(id) if isinstance(id, int) else id.encode()
                value = txn.get(key)
                if value:
                    offset, size = struct.unpack('<QQ', value)
                    # Add the freed region to the free list
                    free_list.append((offset, size))
                    # Remove the key from LMDB
                    txn.delete(key)

            # Merge adjacent free regions
            free_list = self._merge_free_regions(free_list)

            # Check if we can shrink the file
            if free_list:
                last_offset, last_size = free_list[-1]
                if last_offset + last_size == self.current_size:
                    # We can shrink the file
                    self.current_size = last_offset
                    # Truncate the memmap file
                    new_file_size = self.current_size * self.itemsize
                    with open(self.memmap_path, 'rb+') as f:
                        f.truncate(new_file_size)
                    # Re-open the memmap with the new size
                    self.memmap_array = np.memmap(
                        self.memmap_path, dtype=self.dtype, mode='r+', shape=(self.current_size,))
                    # Remove the last free region from the free list
                    free_list.pop()

            # Save the updated free list
            free_list_packed = msgpack.packb(free_list)
            txn.put(b'free_list', free_list_packed)

    def get_data_count(self):
        mmap_size = self.current_size
        with self.env.begin() as txn:
            lmdb_size = txn.stat()['entries']
        print(f"LMDB size: {lmdb_size}, MMAP size: {mmap_size}")
        return mmap_size

    def close(self):
        self.env.close()
