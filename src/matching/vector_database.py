"""
Vector Database Manager
Handles storage and retrieval of embeddings using Chroma and FAISS
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pickle
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")

class VectorDatabase:
    """
    Base class for vector database operations
    """
    
    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
    
    def add_embeddings(self, embeddings: List[np.ndarray], 
                      metadata: List[Dict[str, Any]], 
                      ids: List[str]):
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, 
              top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    def delete(self, ids: List[str]):
        raise NotImplementedError
    
    def get_collection_info(self) -> Dict[str, Any]:
        raise NotImplementedError

class ChromaVectorDB(VectorDatabase):
    """
    ChromaDB implementation for vector storage
    """
    
    def __init__(self, persist_directory: str, collection_name: str = "resume_jd_embeddings"):
        super().__init__(persist_directory)
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Resume and Job Description embeddings"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def add_embeddings(self, embeddings: List[np.ndarray], 
                      metadata: List[Dict[str, Any]], 
                      ids: List[str]):
        """
        Add embeddings to ChromaDB collection
        
        Args:
            embeddings: List of embedding arrays
            metadata: List of metadata dictionaries
            ids: List of unique IDs
        """
        try:
            # Convert numpy arrays to lists for ChromaDB
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            # Add documents to collection
            self.collection.add(
                embeddings=embeddings_list,
                metadatas=metadata,
                ids=ids
            )
            
            logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to ChromaDB: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, 
              top_k: int = 5, 
              filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding array
            top_k: Number of top results to return
            filter_criteria: Optional metadata filter
        
        Returns:
            List of search results with metadata and distances
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_criteria
            )
            
            # Format results
            search_results = []
            if results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    search_results.append({
                        'id': results['ids'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {str(e)}")
            return []
    
    def delete(self, ids: List[str]):
        """Delete embeddings by IDs"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} embeddings from ChromaDB")
        except Exception as e:
            logger.error(f"Failed to delete from ChromaDB: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'type': 'ChromaDB'
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB info: {str(e)}")
            return {'error': str(e)}

class FAISSVectorDB(VectorDatabase):
    """
    FAISS implementation for vector storage
    """
    
    def __init__(self, persist_directory: str, dimension: int, 
                 index_type: str = "FlatIP"):
        super().__init__(persist_directory)
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata_store = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0
        
        self.index_file = self.persist_directory / "faiss_index.idx"
        self.metadata_file = self.persist_directory / "metadata.pkl"
        self.id_mapping_file = self.persist_directory / "id_mapping.json"
        
        self._initialize_index()
        self._load_persistent_data()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            if self.index_type == "FlatIP":
                # Inner product (cosine similarity for normalized vectors)
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "FlatL2":
                # L2 (Euclidean) distance
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IVFFlat":
                # IVF with flat quantizer (for larger datasets)
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
            
            logger.info(f"FAISS index initialized: {self.index_type}, dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {str(e)}")
            raise
    
    def _load_persistent_data(self):
        """Load persistent index and metadata"""
        try:
            # Load FAISS index
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    self.metadata_store = pickle.load(f)
            
            # Load ID mappings
            if self.id_mapping_file.exists():
                with open(self.id_mapping_file, 'r') as f:
                    data = json.load(f)
                    self.id_to_index = data['id_to_index']
                    self.index_to_id = {v: k for k, v in self.id_to_index.items()}
                    self.next_index = data.get('next_index', 0)
            
        except Exception as e:
            logger.warning(f"Failed to load persistent data: {str(e)}")
    
    def _save_persistent_data(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            # Save ID mappings
            with open(self.id_mapping_file, 'w') as f:
                json.dump({
                    'id_to_index': self.id_to_index,
                    'next_index': self.next_index
                }, f)
            
            logger.info("Saved FAISS persistent data")
            
        except Exception as e:
            logger.error(f"Failed to save persistent data: {str(e)}")
    
    def add_embeddings(self, embeddings: List[np.ndarray], 
                      metadata: List[Dict[str, Any]], 
                      ids: List[str]):
        """Add embeddings to FAISS index"""
        try:
            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add to index
            self.index.add(embeddings_array)
            
            # Store metadata and ID mappings
            for i, (embedding, meta, doc_id) in enumerate(zip(embeddings, metadata, ids)):
                current_index = self.next_index + i
                self.metadata_store[current_index] = meta
                self.id_to_index[doc_id] = current_index
                self.index_to_id[current_index] = doc_id
            
            self.next_index += len(embeddings)
            
            # Save to disk
            self._save_persistent_data()
            
            logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to FAISS: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, 
              top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings in FAISS index"""
        try:
            # Reshape query for FAISS
            query_array = query_embedding.reshape(1, -1).astype('float32')
            
            # Search
            distances, indices = self.index.search(query_array, top_k)
            
            # Format results
            search_results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx != -1:  # Valid result
                    doc_id = self.index_to_id.get(idx, f"unknown_{idx}")
                    metadata = self.metadata_store.get(idx, {})
                    
                    # Convert distance to similarity based on index type
                    if self.index_type == "FlatIP":
                        similarity = float(distance)  # Inner product is already similarity
                    else:
                        similarity = 1.0 / (1.0 + float(distance))  # Convert distance to similarity
                    
                    search_results.append({
                        'id': doc_id,
                        'metadata': metadata,
                        'distance': float(distance),
                        'similarity': similarity,
                        'index': int(idx)
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {str(e)}")
            return []
    
    def delete(self, ids: List[str]):
        """Note: FAISS doesn't support deletion. This is a placeholder."""
        logger.warning("FAISS doesn't support deletion. Consider rebuilding index.")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the FAISS index"""
        return {
            'type': 'FAISS',
            'index_type': self.index_type,
            'dimension': self.dimension,
            'count': self.index.ntotal,
            'metadata_count': len(self.metadata_store)
        }

class VectorDBManager:
    """
    Manager class for different vector database implementations
    """
    
    def __init__(self, db_type: str = "chroma", **kwargs):
        """
        Initialize vector database manager
        
        Args:
            db_type: Type of vector database ('chroma', 'faiss')
            **kwargs: Additional parameters for specific database types
        """
        self.db_type = db_type
        self.db = None
        
        if db_type == "chroma":
            persist_dir = kwargs.get('persist_directory', './data/chroma_db')
            collection_name = kwargs.get('collection_name', 'resume_jd_embeddings')
            self.db = ChromaVectorDB(persist_dir, collection_name)
            
        elif db_type == "faiss":
            persist_dir = kwargs.get('persist_directory', './data/faiss_db')
            dimension = kwargs.get('dimension', 384)  # Default for all-MiniLM-L6-v2
            index_type = kwargs.get('index_type', 'FlatIP')
            self.db = FAISSVectorDB(persist_dir, dimension, index_type)
            
        else:
            raise ValueError(f"Unknown database type: {db_type}")
        
        logger.info(f"Vector database manager initialized: {db_type}")
    
    def add_documents(self, documents: List[Dict[str, Any]], 
                     embeddings: List[np.ndarray]):
        """
        Add documents with their embeddings to the vector database
        
        Args:
            documents: List of document dictionaries with metadata
            embeddings: List of corresponding embeddings
        """
        ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
        metadata = [{k: v for k, v in doc.items() if k != 'id'} for doc in documents]
        
        self.db.add_embeddings(embeddings, metadata, ids)
    
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 5, 
                      filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding
            top_k: Number of top results
            filter_criteria: Optional filter (ChromaDB only)
        
        Returns:
            List of similar documents with metadata
        """
        if self.db_type == "chroma" and filter_criteria:
            return self.db.search(query_embedding, top_k, filter_criteria)
        else:
            return self.db.search(query_embedding, top_k)
    
    def get_info(self) -> Dict[str, Any]:
        """Get database information"""
        return self.db.get_collection_info()