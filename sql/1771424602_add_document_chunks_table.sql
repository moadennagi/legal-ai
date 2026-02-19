-- enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- document_chunks tacble

CREATE TABLE IF NOT EXISTS "document_chunks" (
    id SERIAL PRIMARY KEY,
    CONTENT TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(768),
    token_count INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    created_at INTEGER DEFAULT extract(epoch from now())::INTEGER,
    updated_at INTEGER,
    document_id INTEGER NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
    CONSTRAINT uq_chunk_index_per_document UNIQUE (document_id, chunk_index)
);

-- create index for vector similarity
CREATE INDEX IF NOT EXISTS chunks_similarity_idx ON "document_chunks"
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);