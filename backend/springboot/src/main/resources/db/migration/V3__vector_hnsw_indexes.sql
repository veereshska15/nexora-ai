-- ==============================================================================
-- NEXORA AI — V3 VECTOR SEARCH HNSW INDEXES
-- Cosine Distance HNSW Index for Dense Vector Embeddings
-- ==============================================================================

-- Create HNSW index for high-speed approximate nearest neighbor vector similarity search
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx 
ON document_chunks USING hnsw (embedding vector_cosine_ops);
