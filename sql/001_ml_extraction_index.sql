-- Run once before a multi-million-row extraction.
-- CONCURRENTLY avoids blocking battle ingestion while the index is built.
create index concurrently if not exists battles_ml_extraction_cursor_idx
on public.battles (inserted_at, fingerprint);

-- Faster default ML extraction when only ranked/ladder are allowed.
-- This matches the SQL-side mode filter used by `rigged-matchup extract`.
create index concurrently if not exists battles_ml_extraction_ranked_ladder_cursor_idx
on public.battles (inserted_at, fingerprint)
where mode_key in ('ladder', 'ranked');
