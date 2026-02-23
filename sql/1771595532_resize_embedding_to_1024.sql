-- Resize embedding column from 768 to 1024 dimensions for bge-m3.
-- This drops all existing chunks since their 768-dim embeddings are
-- incompatible with the new model and must be regenerated.

-- 1. Drop the old IVFFLAT index (tied to vector(768))
DROP INDEX IF EXISTS chunks_similarity_idx;

-- 2. Clear all existing chunks (embeddings are model-specific)
DELETE FROM document_chunks;

-- 3. Alter the embedding column to the new dimension
ALTER TABLE document_chunks
    ALTER COLUMN embedding TYPE vector(1024);

-- 4. Recreate the IVFFLAT index for vector(1024)
--    lists = sqrt(expected_row_count); update if corpus grows significantly
CREATE INDEX chunks_similarity_idx ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
