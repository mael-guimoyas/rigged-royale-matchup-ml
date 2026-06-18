-- Run once before a multi-million-row extraction.
-- CONCURRENTLY avoids blocking battle ingestion while the index is built.
create index concurrently if not exists battles_ml_extraction_cursor_idx
on public.battles (inserted_at, fingerprint);
