# Graph Report - .  (2026-06-27)

## Corpus Check
- 60 files · ~85,101 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 587 nodes · 1472 edges · 20 communities (16 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 48 edges (avg confidence: 0.75)
- Token cost: 149,579 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `AppConfig` - 50 edges
2. `SymmetricMatchupModel` - 40 edges
3. `train_model()` - 24 edges
4. `load_config()` - 21 edges
5. `binary_metrics()` - 20 edges
6. `collect_from_api()` - 18 edges
7. `StorageClient` - 17 edges
8. `parse_battle_row()` - 17 edges
9. `Deduplicator` - 16 edges
10. `BalancedFrontier` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Per-Battle Win Chance & Matchup Labels UI` --semantically_similar_to--> `Bilinear Oriented Card-vs-Card Interaction`  [INFERRED] [semantically similar]
  .playwright-mcp/page-2026-06-22T10-02-07-706Z.yml → README.md
- `Invalid Player Tag Error Page` --semantically_similar_to--> `Player Tag Seed List (seeds.txt)`  [INFERRED] [semantically similar]
  .playwright-mcp/page-2026-06-22T10-00-43-666Z.yml → seeds.txt
- `main()` --calls--> `Deck`  [INFERRED]
  scripts/segment_diagnostic.py → src/rigged_matchup_ml/domain.py
- `_make_checkpoint()` --calls--> `SymmetricMatchupModel`  [INFERRED]
  tests/test_serve.py → src/rigged_matchup_ml/model.py
- `test_normalize_tag()` --calls--> `normalize_tag()`  [EXTRACTED]
  tests/test_api_collect.py → src/rigged_matchup_ml/api_collect.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Matchup Model Serving Stack** — cloud_run_deploy, cloud_run_inference_server, docker_compose_inference_service, readme_matchup_model, cloud_run_predict_api_key [INFERRED 0.85]
- **Train-to-Serve Flow** — runpod_training_run, readme_train_command, readme_matchup_model, cloud_run_deploy, riggedroyale_app [INFERRED 0.85]

## Communities (20 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (52): Counter, BalancedFrontier, _battle_fingerprint(), ClashRoyaleClient, _clean_env_token(), collect_from_api(), _encode_tag(), _fetch_player() (+44 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (60): Path, attach_prior(), backfill_ranked_segments_command(), benchmark(), collect_api(), diagnose_ranked_segments_command(), drain_db(), evaluate() (+52 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (38): encode_row(), _class_sql(), evaluate_meta(), CardInteractionEncoder, DeckEncoder, PairSummaryEncoder, Positive per-card importance weight ``(B, 8)`` from metadata, or None., Antisymmetric logits guarantee P(A beats B) = 1 - P(B beats A). (+30 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (42): Any, datetime, PriorProvider, diagnose_ranked_segments(), _quoted(), _as_datetime(), canonical_game_id(), _card_role() (+34 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (45): ndarray, benchmark_model(), _calibrated_probabilities(), compute_noise_floor(), _device(), Apply the same per-segment temperature/bias calibration as evaluation., Compare the model against meaningful baselines on the same split.      Produce, Irreducible-error floor from observed per-matchup win-rates.      With binary (+37 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (37): BaseModel, FastAPI, Request, load_bundle(), Load a checkpoint once and attach an eval-ready model under ``model``.      Th, build_response(), CardInteraction, CardInteractions (+29 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (37): DataLoader, LambdaLR, Module, Optimizer, load_card2vec(), Return saved card vectors if present and shape-compatible, else None., matchup_dataloader(), _build_scheduler() (+29 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (32): IterableDataset, RecordBatch, _assemble_batch(), BatchedMatchupIterableDataset, _decode_batch(), _encode_card_values(), encode_rows(), _EncodeContext (+24 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (39): Public Case Board (Most/Least Rigged Rankings), Cloud Run Deployment, Stateless FastAPI Inference Server (POST /predict, GET /health), PREDICT_API_KEY (X-API-Key gate), Default Pipeline Config (config/default.yaml), card2vec_init Config Key, collect_min_trophies Config Key, empirical_app_dir Config Key (../riggedroyale) (+31 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (32): _ability_block(), elixir_for(), _evo_block(), _load_metadata_snapshot(), metadata_for(), metadata_vector_for(), _normalise_metadata(), _parse_ability() (+24 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (25): AppConfig, attach_empirical_prior(), _build_train_matrix(), clamp_prior(), _coverage_rates(), _default_app_dir(), _empty_coverage(), fallback_prior() (+17 more)

### Community 11 - "Community 11"
Cohesion: 0.21
Nodes (15): bucketFromSegment(), buildMatrix(), buildMatrixFromRecords(), BuildMatrixRow, contextFor(), coverageFor(), db, eachInputLine() (+7 more)

### Community 12 - "Community 12"
Cohesion: 0.27
Nodes (6): download_from_storage(), Minimal Supabase Storage REST client (bucket + list/upload/download)., Download training Parquet shards from Supabase Storage into data/raw.      The, Upload local training Parquet shards from data/raw to Supabase Storage., StorageClient, upload_to_storage()

### Community 13 - "Community 13"
Cohesion: 0.67
Nodes (3): main(), Split each prepared split's single data.parquet into N parquet files.  pyarrow, repartition()

## Knowledge Gaps
- **10 isolated node(s):** `BuildMatrixRow`, `ScoreRow`, `db`, `Per-Segment Affine Calibration`, `Cross-Player Battle Deduplication` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AppConfig` connect `Community 10` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 6`, `Community 12`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `SymmetricMatchupModel` connect `Community 2` to `Community 1`, `Community 4`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `collect_from_api()` connect `Community 0` to `Community 1`, `Community 10`, `Community 3`, `Community 12`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `AppConfig` (e.g. with `BalancedFrontier` and `ClashRoyaleClient`) actually correct?**
  _`AppConfig` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `One-off: build the covering index that the Supabase seed query needs.  The see`, `Regenerate the packaged static card metadata snapshot.  The model must not fetch`, `Cycles to charge the evo: 1 for costly cards, 2 for cheap/small ones.` to the rest of the system?**
  _110 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05432595573440644 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06331976481230213 - nodes in this community are weakly interconnected._