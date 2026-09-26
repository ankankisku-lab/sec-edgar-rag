# Prompt: SEC-EDGAR Financial RAG — Architecture & Engineering Report (.docx)

You are a senior ML/RAG engineer and technical writer. You are running inside the repository
`D:\SEC EDGAR RAG` (a Windows machine; Python venv at `.venv`). Your job is to **read this
entire codebase and its recorded results**, then write a **detailed, chronological architecture and
engineering report** and **save it as a Word document**:

    docs/SEC_EDGAR_RAG_Architecture_Report.docx

The reader is a technical reviewer (hiring manager / ML engineer) who has not seen the project.
They should finish the document understanding what was built, why each choice was made, how each
component works internally, what was measured, what failed, and how problems were solved.

---

## 1. Ground rules (non-negotiable)

1. **Evidence only.** Every number you write (metrics, latencies, sizes, counts, versions) must come
   from a file in this repo. After each table or key figure, cite its source file path in small text,
   e.g. *Source: data/eval/results/experiments.csv*. Never invent, round up, or "estimate" a result.
2. **Source-of-truth priority** when files disagree: result CSV/JSON files > `git log` commit messages
   > `README.md` prose > code comments. If you find a disagreement, report it in an
   "Inconsistencies noted" box rather than silently picking one.
3. **Chronology comes from git.** Use `git log --reverse --date=iso --format="%h %ad %s%n%b"` to
   order phases, and quote commit hashes next to each phase.
4. **Read-only.** Do not modify source code, configs, data, or results. Do not re-run experiments,
   indexing, embedding, or any GPU / LLM / network job. The only files you create are the report,
   the figure images you embed in it (under `docs/figures/`), and a helper script if you need one
   (under `docs/tools/`).
5. **No secrets.** Never open or quote `.env`. Refer to keys only by variable name
   (`SEC_USER_AGENT`, `GROQ_API_KEY`, `GOOGLE_API_KEY`, `LLAMA_CLOUD_API_KEY`).
6. **Honesty about status.** Clearly label work that is paused or incomplete (e.g. Ragas scoring,
   scaling stages that have no row in `scaling.csv` yet) and negative results (e.g. HyDE, hybrid
   fusion, the local faithfulness judge). Negative results are findings; present them as such.
7. **Explain, don't just list.** For every design decision state: the problem, the alternatives
   considered, the choice, the evidence, and the trade-off.

---

## 2. What to read (in this order)

1. `README.md` — the running narrative and experiment tables.
2. `git log` (see rule 3) — chronology, commit messages with measured results.
3. `configs/*.yaml` — every tunable and the comments explaining it
   (`ingestion`, `chunking`, `embedding`, `qdrant`, `retrieval`, `generation`, `llm`,
   `evaluation`, `eval_narrative`).
4. Source, module by module:
   - `src/ingestion/` — manifest (Wikipedia → CIK → SEC submissions API), rate-limited SEC client,
     resumable downloader, checkpoints, iXBRL `dei:` verification.
   - `src/parsing/` — HTML → structured elements, section tracking (Part/Item/Note), table
     reconstruction (colspan, `$`/`)` cell merging, header detection), page-artifact removal.
   - `src/chunking/hierarchical.py` — parent/child LlamaIndex `TextNode`s, metadata header,
     deterministic IDs, table row groups with repeated headers.
   - `src/embeddings/embedder.py` — BGE embeddings, caching, token-length checks.
   - `src/retrieval/` — Qdrant index (dense + sparse), BM25 sparse encoder, hybrid fusion (relative,
     RRF), table linearization, table-aware cross-encoder reranker, decomposed retriever, parent store.
   - `src/query/` — sub-question decomposition + router, HyDE.
   - `src/generation/` — Ollama server management, parent expansion, CitationQueryEngine generator,
     `<calc>` arithmetic and number verification, LLM feasibility benchmark.
   - `src/evaluation/` — eval-set builder (v0, v1), retrieval metrics, fusion/rerank sweeps,
     paired bootstrap, generation accuracy, Ragas scorer with judge guards and agreement check.
   - `src/pipeline/scale.py` — Phase 16 scaling stages.
5. `tests/` — what behaviours are pinned by tests (cite them as evidence of correctness).
6. Results:
   - `data/eval/results/experiments.csv` (all retrieval runs; filter by the `dataset` column —
     `retrieval_v0.jsonl` vs `retrieval_v1.jsonl` are different question sets, never mix them in one
     table),
   - `data/eval/results/*_per_query.csv`, `data/eval/results/retrieval_v1/*`,
   - `fusion_sweep.csv`, `rerank_sweep.csv`, `candidate_pools.csv`,
   - `G1..G6_generation.csv`, `G5/G6_ragas_*.csv`,
   - `scaling.csv` and `data/eval/scaling_stages.json`,
   - `data/benchmarks/llm_feasibility.csv`,
   - `data/metadata/{companies,filings,verification}.csv`,
   - `data/eval/retrieval_v0.jsonl`, `retrieval_v1.jsonl` (count question types yourself),
   - checkpoints in `data/checkpoints/*.json` for corpus-level counts (parse/chunk/embed stats).
7. `requirements.txt`, `docker-compose.yml`, `.gitignore`.

You may run **read-only** commands to compute summaries (e.g. pandas over the CSVs, `git log`,
counting lines in JSONL files). Where a paired significance test is quoted, prefer the numbers in the
README/commit messages; if you recompute one, use `src/evaluation/compare.py` logic (paired
bootstrap, 10,000 resamples, 95% CI) and say you recomputed it.

---

## 3. Required document structure

Use real Word heading styles (Heading 1/2/3) so the Table of Contents works.

**Title page** — project title, one-line description, date, repository path, "Generated from the
repository state at commit `<hash>`".

**Table of Contents** — an auto-updating TOC field.

**1. Executive summary** (1 page) — the problem, the final architecture in one paragraph, the 5–8
headline results with their numbers (e.g. retrieval MRR progression V1 → V5, multi-fact all-hit@10,
generation accuracy G3 → G4/G6, unverified-number rate), the hardware/cost constraints (4 GB VRAM
laptop GPU, 16 GB RAM, free-only models), and what is still open.

**1a. Naming key — every short form used in the project** (placed right after the executive summary,
before any results, and repeated compactly as Appendix D). The project labels experiments with short
IDs; a reader must never meet one undefined. Build this key by reading the `version`, `retriever`,
`dataset`, `notes` and option columns of `data/eval/results/experiments.csv`, the result file names
in `data/eval/results/` (including `retrieval_v1/`), the generation eval code and CSVs, the
commit messages, and the README. For **each** ID give: full name, exactly what pipeline/configuration
it denotes (retriever, candidate pool, fusion/α, reranker on/off and input format, decomposition,
HyDE, LLM + prompt version where relevant), which eval set it was measured on (v0 and/or v1), the
phase that introduced it, and where its results live. Cover at least:
- **Retrieval versions:** V1 (dense), V2 (BM25), V3 (hybrid, relative-score fusion α=0.3),
  V4 (BM25 top-60 → linearized BGE reranker), V4-md (V4 ablation: reranker reads markdown tables),
  V4h (V4 with BM25-45 + dense-15 pool), V5 (V4 + sub-question decomposition behind a router),
  V5h (V5 with BM25-45 + plain dense-15 pool — the HyDE control), V6 (V5 with BM25-45 +
  HyDE-dense-15 pool), V1h (dense retrieval with the HyDE embedding), and scaled variants written
  `V2@S50`, `V4@S100`, `V5@Sall`, etc.
- **Generation runs:** G1 (30 numeric questions, eval v0, prompt v1), G2 (40 questions, eval v1,
  prompt v1), G3 (same 40, prompt v2), G4 (same 40, prompt v2 + decomposition), G5 (G3 setup + 24
  narrative questions, answers and contexts saved for Ragas), G6 (G4 setup + 24 narrative).
  Verify each definition against the code/commits before writing it; if a detail differs, trust the
  evidence and note the correction.
- **Scaling stages:** S24, S50, S100, S250, Sall (filings and companies per stage, from
  `data/eval/scaling_stages.json`).
- **Datasets and versions:** eval sets `retrieval_v0` / `retrieval_v1`; parser v1–v4 and chunker
  v1–v3 (what changed in each bump); prompt v1 / v2.
- **Question types:** numeric, yoy, pct_change, cross_quarter, cross_company, narrative.
- **Judges:** `groq-gpt-oss-120b`, `local-llama3.1-8b`.
- **Metrics:** P@3, R@5 (capped), Hit@k, MRR@10, nDCG@10, all-hit@k, fact_recall@10 — brief
  definitions here, full definitions in the methodology chapter.
Present it as a table (ID | stands for | exact configuration | eval set | phase | results file). If
you find an ID in the results that you cannot define from the evidence, list it as
"undefined — needs author confirmation" rather than guessing.

Throughout the report, **spell out an ID in parentheses on its first use in every chapter**, e.g.
"V4 (BM25 top-60 → linearized BGE reranker)", "G6 (decomposition, 64 questions incl. narrative)",
and add the ID key's column headers to every results table that uses IDs.

**2. Constraints and principles** — hardware limits and how they shaped every choice; the
"measure every component against a baseline" rule; the free-models-only policy; data stored on D:.

**3. System architecture**
- 3.1 Offline ingestion/indexing pipeline (SEC → manifest → download → verify → parse → chunk →
  embed → Qdrant + parent store). Include a **diagram** (PNG you generate, see §5).
- 3.2 Online query pipeline (router → decomposition → per-sub-question BM25 → linearized BGE
  reranker → interleave → parent expansion → CitationQueryEngine → Qwen → `<calc>` → number
  verification). Include a **diagram**.
- 3.3 Data model: element types, parent/child nodes, metadata fields shown to embedder vs LLM,
  deterministic IDs, Qdrant collection layout (named dense + sparse vectors, payload indexes).
- 3.4 Repository map: one line per module.

**4. Models and components — what, why, and how they work internally.** One subsection per item,
each with: role, exact model/version (from configs/requirements), footprint measured on this machine,
why it was chosen over alternatives, and **the internal mechanism that made it a good fit**. Cover at
least:
- `BAAI/bge-small-en-v1.5` (bi-encoder: separate query/passage encoding, CLS pooling, normalized
  cosine, query instruction prefix; asymmetric query vs passage embedding; 512-token limit; fp16).
- BM25 via fastembed `Qdrant/bm25` + Qdrant IDF modifier (TF saturation `k1`, length normalization
  `b` with measured `avg_len`, why IDF is computed server-side, stemming/stopwords).
- `BAAI/bge-reranker-base` cross-encoder (joint query–passage attention vs bi-encoder; why raw
  logits instead of sigmoid; why table linearization; fp16; batch size choice).
- Qdrant (HNSW for dense, inverted index for sparse, payload indexes, on-disk payload, named volume
  on Windows) and the SQLite parent store.
- LlamaIndex components actually used (TextNode relationships, QdrantVectorStore hybrid mode,
  SentenceSplitter, SentenceTransformerRerank subclass, CitationQueryEngine, HyDEQueryTransform,
  Ollama LLM) and where the project deliberately bypassed LlamaIndex defaults and why.
- Qwen3-4B-Instruct-2507 (Q4_K_M) via Ollama — why a local 4B model; the feasibility benchmark vs
  Llama-3.2-3B and Phi-4-mini; Q4_K_M quantization, q8_0 KV cache, flash attention, CPU/GPU layer
  split under ~2.25 GB usable VRAM; why arithmetic is delegated to Python.
- Judges: Groq `openai/gpt-oss-120b` (free plan, rate/daily limits, guard design) and local
  `llama3.1:8b` (schema-constrained decoding; agreement results; why it was rejected for faithfulness).
- Ragas 0.4 collections metrics (how faithfulness and answer relevancy are computed; the refusal
  caveat; the terse-answer relevancy caveat).

**5. Chronological phase-by-phase account** — the core of the report. For **each** phase in commit
order (including phases done out of the original order, e.g. decomposition before Ragas, and
unplanned work such as the local-LLM feasibility benchmark): *Goal → Design → What was built (files)
→ Experiments & measurements (tables) → Problems found & how they were fixed → Checkpoint verdict →
Commit(s)*. The phases include, at minimum: 0–2 setup & corpus acquisition; 3 parsing; 4 chunking;
5 embeddings; 6 Qdrant; 7 eval set v0 + dense baseline; 8–9 BM25 + hybrid; 10 reranking;
local-LLM feasibility; 11 generation; 12 eval set v1; 14 decomposition; 13 Ragas (paused);
15 HyDE; 16 scaling.

**6. Experiment ledger** — one consolidated section:
- Retrieval, eval v0 and eval v1 as **separate** tables (P@3, R@5, MRR@10, Hit@10, all-hit@10,
  per-type breakdown, p50 latency), including ablations/controls (V4-md, V4h, V5h, V1h, V6).
- Sweeps: fusion (relative α vs RRF, dev vs test), rerank pool × linearization, candidate-pool recall.
- Significance: paired bootstrap CIs for the key comparisons; McNemar for G3 → G4.
- Generation: G1…G6 (correct overall, correct given facts in context, by type, refusals,
  unverified numbers, citation rate, latency).
- LLM feasibility grid (VRAM/on-GPU, tok/s, correctness) and the co-residency test.
- Judge agreement (120B vs 8B).
- Scaling curve (whatever stages exist in `scaling.csv`): quality vs corpus size, index size,
  memory, indexing time.
For every table: 2–4 sentences of interpretation.

**7. Bottlenecks, bugs and fixes** — a table *and* short narratives, chronologically: symptom → how
it was diagnosed (the measurement that exposed it) → root cause → fix → measured effect. Mine the
commit messages and README for these; examples to verify and include if present: Windows
`localhost` IPv6 stall (~2 s → ~8 ms per Qdrant request), checkpoint rename blocked by file locks,
C:-drive caches redirected to D:, nested iXBRL tags, `<br>`/block text concatenation, spanned headers
merged into the label column, Microsoft year-row headers, split/continuation tables, oversized
footnote chunks, reranker sigmoid saturation, markdown tables vs prose-trained cross-encoder,
orphaned `llama-server.exe` holding VRAM (15× slower prompt processing), reranker batch size,
compound questions breaking BM25, eval-question generation bugs (cash-flow change rows, group
labels), period ambiguity, dependency conflicts (ragas / langchain-community / instructor / openai),
async event-loop reuse, per-minute vs daily free-tier limits, accidental data commits to git.

**8. Evaluation methodology** — how eval sets v0/v1 were built (fact extraction, period validation,
gold groups, narrative relevance rules), the P@3 ceiling, metric definitions, dev/test split for
selecting settings, why deterministic gold is preferred over LLM judging for retrieval, and known
limitations of the eval set.

**9. Results in perspective** — what the numbers do and don't show; negative results (dense, hybrid,
HyDE, local faithfulness judge) and why they happened mechanistically; remaining failure modes.

**10. Open items and next steps** — paused Ragas plan, remaining phases (numerical-hallucination
evaluation, quantization, Phoenix tracing, FastAPI, Docker), known issues (period ambiguity, filers
with non-standard sectioning), with the evidence for each.

**Appendix A — Reproduction commands** (from README/configs; mark GPU/LLM/network steps).
**Appendix B — Configuration reference** (key parameters with their values and comments).
**Appendix C — Glossary** (MRR, all-hit@k, RRF, cross-encoder, HyDE, KV cache, Q4_K_M, iXBRL, etc.).
**Appendix D — Run-ID quick reference** (the naming key from §1a, one line per ID, alphabetical).

---

## 4. Writing style

- Precise and technical, but readable: short paragraphs, active voice, no marketing language.
- Define each acronym on first use. Use the project's own names (V1–V6, G1–G6, S24–Sall).
- Tables for numbers; prose for reasoning. Right-align numeric columns; 3 decimals for rates.
- Target length: roughly 25–45 pages including tables and figures. Depth over padding.

---

## 5. Building the .docx

1. Use `python-docx` from the project venv (`.venv\Scripts\python`). If it is missing, install it
   into the venv with `pip install python-docx` (keep pip's cache on D:, e.g.
   `PIP_CACHE_DIR="D:/SEC EDGAR RAG/.cache/pip"`). Also use `matplotlib` for figures (install the
   same way if missing).
2. Generate figures as PNG in `docs/figures/` and embed them with numbered captions:
   - offline pipeline diagram and online query pipeline diagram (boxes and arrows drawn with
     matplotlib; clean, legible at page width),
   - retrieval MRR@10 by version (eval v1) as a bar chart,
   - all-hit@10 by question type for V1/V2/V4/V5,
   - generation accuracy G3 vs G4 (or G5 vs G6) by question type,
   - scaling curve (MRR/P@3 vs number of filings) if `scaling.csv` has ≥2 stages.
   Every chart's data must come from the result files; put the source path in the caption.
3. Formatting: Heading 1/2/3 styles; a TOC field (`TOC \o "1-3" \h \z \u`) with a note that Word
   updates it on open (F9); tables with a bold shaded header row and light borders; code, paths and
   commands in a monospace font; page numbers in the footer; the document title in the header;
   11-pt body font.
4. Keep the generator script at `docs/tools/build_report.py` so the report can be regenerated.
5. Save to `docs/SEC_EDGAR_RAG_Architecture_Report.docx`.

---

## 6. Final verification (do this before you finish)

- Re-open the .docx with python-docx and print: number of headings per level, number of tables,
  number of figures, and approximate word count.
- Spot-check at least 10 numbers in the document against their source files and list them
  (value, section, source file) in your final message.
- Scan the document text for every run-ID pattern (`V\d[\w-]*`, `V\d@S\w+`, `G\d`, `S\d+|Sall`)
  and confirm each one appears in the naming key; list any that don't and add them.
- Confirm no content from `.env` appears anywhere (search the document text for `gsk_`, `llx-`,
  `AQ.`, and the SEC user-agent email).
- In your final message: the output path, the verification results, any inconsistencies you found
  between sources, and anything you could not document because the evidence was missing.
