# SEC-EDGAR Financial RAG & Evaluation Engine

RAG over Nasdaq-100 10-K / 10-Q filings with hybrid retrieval (BM25 + dense in Qdrant),
BGE cross-encoder reranking, HyDE / sub-question decomposition, and Ragas evaluation.

## Setup

```powershell
C:\Users\ACER\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # then set SEC_USER_AGENT="Name email@domain"
```

## Corpus pipeline

```powershell
# 1. Wikipedia Nasdaq-100 -> CIK -> latest 2x 10-K + 6x 10-Q per company
.venv\Scripts\python -m src.ingestion.metadata
#    -> data/metadata/companies.csv, data/metadata/filings.csv (the manifest)

# 2. Download primary documents (resumable; re-run to retry failures)
.venv\Scripts\python -m src.ingestion.downloader --tickers AAPL MSFT NVDA
.venv\Scripts\python -m src.ingestion.downloader            # full manifest
#    -> data/raw/<TICKER>/<filing_id>.htm + .json metadata sidecar

# 3. Check each file's iXBRL cover tags (form, period end, CIK) against the manifest
.venv\Scripts\python -m src.ingestion.verify
#    -> data/metadata/verification.csv (also records fiscal year / fiscal quarter)
```

## Parsing

```powershell
.venv\Scripts\python -m src.parsing.parser --tickers AAPL MSFT NVDA   # or no args for all
#    -> data/processed/<TICKER>/<filing_id>.json
```

Each filing becomes an ordered list of elements (`heading`, `paragraph`, `table`,
`footnote`) labelled with `part`, `item`, `section`, `note` and `subsection`.
Tables are rebuilt into clean markdown grids (currency/parenthesis cells merged,
period headers aligned, continuation tables inherit their headers) and carry a
caption, units and linked footnotes. Page headers/footers and the table of
contents are dropped. `stats.section_mode` is `titles` for the few filers that
don't use standard "Item N" headings (Intel, Honeywell 10-K); their section
labels are lower confidence.

## Chunking (LlamaIndex parent-child nodes)

```powershell
.venv\Scripts\python -m src.chunking.hierarchical      # settings in configs/chunking.yaml
#    -> data/chunks/<TICKER>/<filing_id>.jsonl  (LlamaIndex TextNodes, parents then children)
```

- **Parents** (LLM context): heading-delimited text groups (<=1024 tokens) and whole
  tables with caption, units and footnotes (<=2048 tokens, split with repeated headers).
- **Children** (retrieval units, ~256 tokens): sentence-split text, or table row
  groups that repeat the period header rows so every number keeps its column label.
- Each child's embedding text is prefixed with company, form, fiscal period, heading
  path, table caption and units; IDs are deterministic (`uuid5(accession/p/c)`).
- Full corpus: ~240k children, ~103k parents; every child <=512 tokens incl. metadata.

## Embeddings and Qdrant index

```powershell
# torch must come from the CUDA index first (see requirements.txt)
.venv\Scripts\python -m src.embeddings.embedder --tickers AAPL MSFT NVDA --smoke-test
#    -> data/embeddings/bge-small-en-v1.5/<TICKER>/<filing_id>.npz (cached vectors)

docker compose up -d                      # Qdrant v1.19.1, dashboard: http://127.0.0.1:6333/dashboard
.venv\Scripts\python -m src.retrieval.qdrant_index --tickers AAPL MSFT NVDA
#    -> Qdrant collection sec_filings_bge_small (children: dense vector + payload)
#    -> data/index/parents.sqlite (parents, fetched by id for context expansion)
```

- Collection has a named dense vector `text-dense` and a reserved sparse vector
  `text-sparse-new` (IDF modifier, for BM25) in LlamaIndex's naming, so
  `QdrantVectorStore` reads it directly; keyword/datetime payload indexes on ticker,
  form, fiscal year/period, section, content type, dates.
- Re-indexing is idempotent (points deleted by accession before upsert).
- Use `127.0.0.1`, not `localhost`: on Windows `localhost` tries IPv6 first and each
  request stalls ~2 s (indexing 24 filings: 3m46s -> 21s).

## Retrieval evaluation

```powershell
.venv\Scripts\python -m src.evaluation.dataset --tickers AAPL MSFT NVDA     # -> data/eval/retrieval_v0.jsonl
.venv\Scripts\python -m src.evaluation.retrieval --retriever dense --version V1
#    -> data/eval/results/experiments.csv (one row per experiment) + per-query CSV and traces
```

`retrieval_v0` (24 filings, 164 questions), ground truth is deterministic and auditable:
- **140 numeric** questions generated from primary-statement rows (current-period column only;
  the period text is validated against the filing's period end). Relevant = every child that
  contains that row with that value, or a text chunk stating the value with the line item.
- **24 narrative** questions from `configs/eval_narrative.yaml`, each with an explicit
  relevance rule (filing scope, heading, required keywords); 2-28 relevant chunks each.

| Version | Retrieval | Reranker | P@3 | R@5 | MRR@10 | Hit@10 | numeric P@3 | narrative P@3 | p50 latency |
|---|---|---|---|---|---|---|---|---|---|
| V1 | Dense (bge-small) | - | 0.091 | 0.109 | 0.159 | 0.250 | 0.026 | 0.472 | 23 ms |
| V2 | BM25 (Qdrant sparse) | - | **0.266** | **0.509** | **0.497** | 0.768 | **0.245** | 0.389 | 14 ms |
| V3 | Hybrid: dense30 + BM25-30, relative fusion alpha=0.3 | - | 0.258 | 0.481 | 0.477 | **0.780** | 0.229 | 0.431 | 31 ms |

Fusion was chosen on the dev half (`python -m src.evaluation.fusion_sweep`): relative score
fusion with alpha 0.3 (dev MRR 0.508) beat alpha 0.5 (0.365), 0.7 (0.243) and RRF (0.292) --
RRF gives dense's mostly-wrong numeric candidates an equal vote. For top-10 ranking, hybrid
ties BM25 on this 85%-numeric set; its value is the reranker's candidate pool
(`fusion_sweep --pools`): dense top-60 contains a relevant chunk for 57% of questions,
BM25 top-60 for 97%, and only the dense+BM25 union reaches 100% on narrative questions.

V1 finding: for numeric questions the top-3 is always the right company and 48% the right
filing, but only 6% of those chunks contain the asked line item -- a table chunk's embedding
is dominated by its headers and numbers, so one row label is diluted. Exact-term matching
(BM25) and query-chunk cross-attention (reranker) target exactly this.

## Local LLM feasibility (4 GB VRAM)

```powershell
# Ollama portable v0.34.4 unzipped to tools\ollama (not the C: installer); models in models\ollama
.venv\Scripts\python -m src.generation.llm_feasibility --pull
.venv\Scripts\python -m src.generation.llm_feasibility             # 3 models x 3 contexts x 2 KV types
.venv\Scripts\python -m src.generation.llm_feasibility --coresident qwen3-4b --context 8192
```

One real RAG prompt (Apple FY2025 Q3 parents; answer: operating income $28,202M vs $25,352M).
All candidates are Q4_K_M, flash attention on, one parallel slot.

| model (Q4_K_M) | on GPU | gen tok/s 4k / 8k (q8 KV) | co-resident 8k* | right values | wrong numbers |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | ~2.25 of 3.1-3.4 GB | 17.0 / 7.3 | 8.2 tok/s, 21 s | 6/6 | 0/6 |
| Llama-3.2-3B-Instruct | 2.2 GB (fully at 4k) | 61.0 / 31.4 | 32.0 tok/s, 15 s | 5/6 | 2/6 |
| Phi-4-mini-instruct | ~2.25 of 3.2-3.5 GB | 25.7 / 13.1 | - | 4/6 | 4/6 |

\* embedder + bge-reranker-base (fp16, 927 MiB) loaded first, as in the live API; total GPU 3.3/4.0 GB.
Windows reserves ~0.8 GB of the GPU and llama.cpp keeps ~1 GB free, so every model gets ~2.25 GB of
VRAM and spills the rest to CPU; an 8-bit KV cache cuts memory and speeds up every configuration.
This is feasibility only: quality is decided on the evaluation set in the generation phase.

Filing IDs are `<TICKER>_<FORM>_<period_of_report>`, e.g. `AAPL_10Q_2025-06-28`.
Foreign private issuers (ARM, ASML, PDD, ...) file 20-F/40-F and are excluded by design.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```
