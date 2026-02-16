-- table creation
CREATE TABLE IF NOT EXISTS "sources" (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    url VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS "documents" (
    id SERIAL PRIMARY KEY,
    url VARCHAR(255) NOT NULL,
    number VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER DEFAULT extract(epoch from now())::INTEGER,
    updated_at INTEGER NULL,
    source_id INTEGER NOT NULL,
    FOREIGN KEY(source_id) REFERENCES sources(id),
    CONSTRAINT uq_number_per_source UNIQUE ("source_id", "number")
);

CREATE TABLE IF NOT EXISTS "tasks" (
    id SERIAL PRIMARY KEY,
    status VARCHAR(255) CHECK(status IN ('failed', 'succeeded', 'in_progress')) NOT NULL,
    created_at INTEGER DEFAULT extract(epoch from now())::INTEGER,
    updated_at INTEGER NULL,
    type VARCHAR(255) CHECK(type IN ('crawling', 'download')) NOT NULL,
    source_id INTEGER NOT NULL,
    parent_id INTEGER NULL,
    FOREIGN KEY(source_id) REFERENCES sources(id),
    FOREIGN KEY(parent_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS "targets" (
    id SERIAL PRIMARY KEY,
    url VARCHAR(255) NOT NULL,
    number VARCHAR(255) NOT NULL,
    created_at INTEGER DEFAULT extract(epoch from now())::INTEGER,
    updated_at INTEGER,
    claimed_at INTEGER,
    task_id INTEGER,
    source_id INTEGER,
    document_id INTEGER,
    FOREIGN KEY(document_id) REFERENCES documents(id),
    FOREIGN KEY(task_id) REFERENCES tasks(id),
    FOREIGN KEY(source_id) REFERENCES sources(id)
);