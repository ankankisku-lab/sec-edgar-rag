# How this project was built

A step-by-step record from the first download to the finished chat UI. For each step: what was
built, what went wrong, how it was fixed, and the measurement that decided whether the fix stayed.
Numbers come from the [architecture report](SEC_EDGAR_RAG_Architecture_Report.docx) and the README.

## The loop behind every step

Every component had to beat the version before it on a fixed evaluation set before it was kept.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 300
---
flowchart LR
    build["Build one change<br/>(one component or one fix)"] --> measure["Measure it on a fixed eval set<br/>deterministic answer keys"]
    measure --> test{"Better than before?<br/>paired bootstrap 95% CI,<br/>McNemar for answers"}
    test -->|"yes"| keep["Keep it<br/>new baseline"]
    test -->|"no"| reject["Reject it<br/>and keep the result on record"]
    keep --> diagnose["Read the failures<br/>(traces, per-question files)"]
    reject --> diagnose
    diagnose --> build

    classDef step fill:#2F5496,stroke:#1F3864,color:#fff
    classDef good fill:#3F7D3A,stroke:#2A5427,color:#fff
    classDef bad fill:#9CA3AF,stroke:#6B7280,color:#fff
    class build,measure,diagnose step
    class keep good
    class reject bad
```

**How to read the diagrams below:** each step is one row, from left to right:

| Box | Meaning |
|---|---|
| **⚠ red** | the problem that came up |
| **🔧 green** | the fix |
| **✅ blue** | the test or measurement that confirmed it |
| **✗ grey** | an idea that was measured and rejected |

Steps are numbered in the order they happened.

## Part 1: from SEC EDGAR to a searchable index

```mermaid
---
config:
  flowchart:
    wrappingWidth: 260
---
flowchart TB
    subgraph s1["1 · Corpus: manifest, download, verification"]
        direction LR
        p1["⚠ Windows file locks refused the checkpoint rename, so downloads crashed mid-run<br/>⚠ nested iXBRL tags produced false period mismatches<br/>⚠ 7 foreign issuers file 20-F/40-F, not 10-K/10-Q"]
        f1["🔧 retry the atomic rename<br/>🔧 nesting-aware tag extraction<br/>🔧 exclude foreign issuers by design"]
        t1["✅ 726 filings from 93 companies<br/>✅ 726/726 match their own cover tags"]
        p1 --> f1 --> t1
    end
    subgraph s2["2 · Parser: HTML into sections and clean tables"]
        direction LR
        p2["⚠ words glued together ('June 28,2025')<br/>⚠ spanned period headers merged into the label column<br/>⚠ split tables lost their period headers<br/>⚠ missing Item headings; Intel uses font size, not bold"]
        f2["🔧 spaces at block boundaries<br/>🔧 header rows never merge into labels<br/>🔧 continuation tables inherit headers<br/>🔧 title fallback and font-size headings"]
        t2["✅ 44,870 tables, 10,725 footnotes<br/>✅ 716/726 filings with standard Items<br/>✅ parser unit tests"]
        p2 --> f2 --> t2
    end
    subgraph s3["3 · Parent-child chunking"]
        direction LR
        p3["⚠ children labelled with the wrong subheading<br/>⚠ footnote chunks over the 512-token embedder limit"]
        f3["🔧 label each child at its midpoint<br/>🔧 sentence-split long footnotes<br/>🔧 table row groups repeat the period header"]
        t3["✅ 240,832 children, max 490 tokens<br/>✅ every table child carries its periods"]
        p3 --> f3 --> t3
    end
    subgraph s4["4 · Dense embeddings (bge-small, 4 GB GPU)"]
        direction LR
        p4["⚠ model caches landed on C:<br/>⚠ one vector per table dilutes the single row asked for"]
        f4["🔧 cache paths redirected to D:<br/>🔧 kept as a baseline to beat"]
        t4["✅ right company in the top 3 every time, but only 6% of chunks contain the asked line item"]
        p4 --> f4 --> t4
    end
    subgraph s5["5 · Qdrant index and parent store"]
        direction LR
        p5["⚠ every request to 'localhost' took ~2 s (Windows tries IPv6 first)"]
        f5["🔧 connect to 127.0.0.1<br/>🔧 parents in SQLite, fetched by id"]
        t5["✅ ~8 ms per request; indexing 3 min 46 s → 21 s<br/>✅ top-10 identical to brute-force search"]
        p5 --> f5 --> t5
    end
    s1 --> s2 --> s3 --> s4 --> s5

    classDef bad fill:#FDECEA,stroke:#B42318,color:#7A1810
    classDef fix fill:#E7F5EC,stroke:#3F7D3A,color:#1F4D1C
    classDef ok fill:#E3EDFB,stroke:#2F5496,color:#1F3864
    class p1,p2,p3,p4,p5 bad
    class f1,f2,f3,f4,f5 fix
    class t1,t2,t3,t4,t5 ok
```

## Part 2: finding what actually works

Retrieval quality on eval set v1 is MRR@10 (the rank of the first right result); all-hit@10 means
every fact a question needs is in the top 10.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 260
---
flowchart TB
    subgraph s6["6 · Which LLM fits in 4 GB of VRAM?"]
        direction LR
        p6["⚠ Windows and llama.cpp leave only ~2.25 GB for the model"]
        f6["🔧 benchmark 3 local 4-bit models on a real filing prompt"]
        t6["✅ Qwen3-4B: 6/6 right, 0 invented numbers<br/>(Llama-3.2-3B faster but 2/6 with wrong numbers)"]
        p6 --> f6 --> t6
    end
    subgraph s7["7 · Evaluation set v0 and the dense baseline"]
        direction LR
        p7["⚠ no trustworthy way to score retrieval"]
        f7["🔧 164 questions from statement rows, periods checked against the filing"]
        t7["✅ dense retrieval MRR 0.153 on v0: the baseline"]
        p7 --> f7 --> t7
    end
    subgraph s8["8 · BM25 and hybrid search"]
        direction LR
        p8["⚠ dense search misses exact line items like 'operating income'"]
        f8["🔧 BM25 sparse vectors in Qdrant"]
        t8["✅ BM25: MRR +0.375 (significant)"]
        r8["✗ hybrid fusion, tuned on a dev split: no gain over BM25, rejected"]
        p8 --> f8 --> t8
        f8 -.-> r8
    end
    subgraph s9["9 · Cross-encoder reranking"]
        direction LR
        p9["⚠ scores saturated at 0.999, so ties<br/>⚠ the reranker preferred prose about a metric over the table row containing it<br/>⚠ Microsoft year rows not detected as headers"]
        f9["🔧 rank on raw logits<br/>🔧 show tables to it as sentences<br/>🔧 parser v4 header detection"]
        t9["✅ table-as-sentences alone: MRR +0.323<br/>✅ MRR 0.428 → 0.614 on v1"]
        p9 --> f9 --> t9
    end
    subgraph s10["10 · Grounded generation"]
        direction LR
        p10["⚠ prompt processing fell to ~70 tok/s (orphaned LLM runners held VRAM)<br/>⚠ small models make arithmetic slips"]
        f10["🔧 kill stale runners; load reranker before the LLM<br/>🔧 LLM writes #lt;calc#gt;, Python computes; every number checked against sources"]
        t10["✅ 80% correct on 30 numeric questions<br/>✅ 0% answers with an unsupported number"]
        p10 --> f10 --> t10
    end
    subgraph s11["11 · Evaluation set v1"]
        direction LR
        p11["⚠ v0 had no comparison questions<br/>⚠ some generated questions were flawed"]
        f11["🔧 236 questions in 6 types, one gold group per fact<br/>🔧 prompt v2 (state differences, no stray refusals)"]
        t11["✅ exposed the biggest gap: comparisons at 0.000 all-hit<br/>✅ answers 40% → 52.5%"]
        p11 --> f11 --> t11
    end
    subgraph s12["12 · Sub-question decomposition"]
        direction LR
        p12["⚠ 'compare X and Y' pulls MD&A comparison text, not the two rows<br/>⚠ the LLM reworded single questions<br/>⚠ an extra LLM call on every question (2.3 s)"]
        f12["🔧 split into one lookup per company / period<br/>🔧 keep the original wording for single lookups<br/>🔧 a regex router decides when to split"]
        t12["✅ all-hit 0.792 → 0.932<br/>✅ answers 52.5% → 87.5% (McNemar p = 0.0005)<br/>✅ p50 back to 0.5 s"]
        p12 --> f12 --> t12
    end
    subgraph s13["13 · Ragas with free judges"]
        direction LR
        p13["⚠ dependency conflicts, async errors<br/>⚠ free-tier token limits<br/>⚠ the local 8B judge echoed its schema"]
        f13["🔧 pinned versions, one event loop, cache and clean stops<br/>🔧 schema-constrained decoding"]
        t13["✅ 120B judge: faithfulness 0.890 on 27 answers<br/>✅ 8B valid for relevancy only (Pearson 0.98 vs 0.44)<br/>⏸ paused at the free-tier limit"]
        p13 --> f13 --> t13
    end
    subgraph s14["14 · HyDE"]
        direction LR
        p14["⚠ would a hypothetical answer passage improve search?"]
        r14["✗ tested with controls: no significant gain, +3 s per query, rejected"]
        p14 -.-> r14
    end
    subgraph s15["15 · Scaling to all 726 filings"]
        direction LR
        p15["⚠ every added filing is a potential distractor"]
        f15["🔧 grow in stages 24 → 56 → 104 → 254 → 726, re-measuring each time"]
        t15["✅ final pipeline MRR −0.010 (not significant) at 30× the data<br/>✅ search p50 stays ~0.5 s"]
        p15 --> f15 --> t15
    end
    s6 --> s7 --> s8 --> s9 --> s10 --> s11 --> s12 --> s13 --> s14 --> s15

    classDef bad fill:#FDECEA,stroke:#B42318,color:#7A1810
    classDef fix fill:#E7F5EC,stroke:#3F7D3A,color:#1F4D1C
    classDef ok fill:#E3EDFB,stroke:#2F5496,color:#1F3864
    classDef no fill:#F3F4F6,stroke:#9CA3AF,color:#4B5563,stroke-dasharray: 4 3
    class p6,p7,p8,p9,p10,p11,p12,p13,p14,p15 bad
    class f6,f7,f8,f9,f10,f11,f12,f13,f15 fix
    class t6,t7,t8,t9,t10,t11,t12,t13,t15 ok
    class r8,r14 no
```

## Part 3: shipping it as a service

```mermaid
---
config:
  flowchart:
    wrappingWidth: 260
---
flowchart TB
    subgraph s16["16 · Tracing with Arize Phoenix"]
        direction LR
        p16["⚠ phoenix.otel.register() broke with the new exporter<br/>⚠ OpenTelemetry's 128-attribute cap dropped the retrieved documents"]
        f16["🔧 configure the OpenTelemetry SDK directly<br/>🔧 raise the span limit; custom spans per stage"]
        t16["✅ per-stage latency: generation is ~95% of the time, retrieval under 1 s"]
        p16 --> f16 --> t16
    end
    subgraph s17["17 · FastAPI service and chat UI"]
        direction LR
        p17["⚠ one GPU cannot run two pipelines at once<br/>⚠ seen live: vague periods ('fiscal 2025') pick the wrong filing"]
        f17["🔧 one pipeline at a time, at most 4 queued, then 503<br/>🔧 health and metrics endpoints; chat UI at GET /"]
        t17["✅ /health answers in 73 ms while an answer is generating<br/>📝 period resolution recorded as a known limit"]
        p17 --> f17 --> t17
    end
    subgraph s18["18 · Docker: the full stack on the GPU"]
        direction LR
        p18["⚠ SQLite locking fails on a Windows bind mount<br/>⚠ offline model check broke on Windows paths<br/>⚠ an 8 GiB prompt cache got the LLM killed for memory<br/>⚠ a 63 s answer after the model unloaded when idle"]
        f18["🔧 open the parent store read-only<br/>🔧 load the cached model snapshot directly<br/>🔧 turn the prompt cache off<br/>🔧 keep the model loaded"]
        t18["✅ warm single-fact answer in 3.9 s<br/>✅ memory stays flat over many questions"]
        p18 --> f18 --> t18
    end
    s16 --> s17 --> s18

    classDef bad fill:#FDECEA,stroke:#B42318,color:#7A1810
    classDef fix fill:#E7F5EC,stroke:#3F7D3A,color:#1F4D1C
    classDef ok fill:#E3EDFB,stroke:#2F5496,color:#1F3864
    class p16,p17,p18 bad
    class f16,f17,f18 fix
    class t16,t17,t18 ok
```

## Part 4: proving the numbers

```mermaid
---
config:
  flowchart:
    wrappingWidth: 260
---
flowchart TB
    subgraph s19["19 · Numerical-hallucination evaluation"]
        direction LR
        p19["⚠ the number check only asks 'is this number anywhere in the sources?'"]
        f19["🔧 plant known errors in correct answers<br/>🔧 trace every wrong number to its table cell<br/>🔧 re-run without #lt;calc#gt;"]
        t19["✅ invented numbers caught 97.5%; wrong-cell numbers 0%<br/>✅ without #lt;calc#gt;: same accuracy, but 60% of correct answers flagged, so #lt;calc#gt; stays"]
        p19 --> f19 --> t19
    end
    subgraph s20["20 · Citation view: show where each number came from"]
        direction LR
        p20["⚠ a citation named a filing, not the spot<br/>⚠ 30% of cells had no column header (period labels in the first column)<br/>⚠ unit rows leaked into headers"]
        f20["🔧 parse each source back into tables; trace every number to its cell<br/>🔧 period labels and year rows count as headers"]
        t20["✅ 97% of cells get a column header<br/>✅ checked in a real browser, light and dark"]
        p20 --> f20 --> t20
    end
    subgraph s21["21 · Highlight in the original SEC filing, no re-parse"]
        direction LR
        p21["⚠ the same fact appears in several statements<br/>⚠ MD&A tables are untagged, so the statement got highlighted instead<br/>⚠ 18 s lookups on large filings<br/>⚠ all 726 downloads carry an injected sec.gov script"]
        f21["🔧 match iXBRL facts and untagged cells, rank by row label, table overlap, period, Item<br/>🔧 index each filing's text once<br/>🔧 strip scripts; no-script CSP"]
        t21["✅ 96.8% of 434 numbers land on exactly one spot<br/>✅ all 277 table values in the quoted table<br/>✅ 18 s → 2 s cold, 0.02 s cached"]
        p21 --> f21 --> t21
    end
    subgraph s22["22 · 63 tested example questions"]
        direction LR
        p22["⚠ 6 of 80 answers passed 'every number found in a source' but were wrong: year-to-date for a quarter, a segment for the company total"]
        f22["🔧 check every stated number against the filing's iXBRL facts: concept, period, segment"]
        t22["✅ 63 verified questions in the README<br/>🔎 exposed 2 bugs in the number check"]
        p22 --> f22 --> t22
    end
    subgraph s23["23 · Fixing the two bugs"]
        direction LR
        p23["⚠ amounts like $2,002 were skipped as years<br/>⚠ segment figures passed as company totals"]
        f23["🔧 a $ or thousands comma always marks an amount<br/>🔧 warn with the segment name and the company total"]
        t23["✅ invented-number detection 96.5% → 97.5%<br/>✅ segment warning: 0 false alarms on 434 numbers"]
        p23 --> f23 --> t23
    end
    s19 --> s20 --> s21 --> s22 --> s23

    classDef bad fill:#FDECEA,stroke:#B42318,color:#7A1810
    classDef fix fill:#E7F5EC,stroke:#3F7D3A,color:#1F4D1C
    classDef ok fill:#E3EDFB,stroke:#2F5496,color:#1F3864
    class p19,p20,p21,p22,p23 bad
    class f19,f20,f21,f22,f23 fix
    class t19,t20,t21,t22,t23 ok
```

## Still open

- **Ragas:** paused at 32 judged answers because of free-tier limits. It resumes from cache.
- **Vector quantization (Phase 18):** not run yet.
- **Period-aware number check:** compare the period and filing of each quoted cell with the period
  the question asks for. This is the gap step 19 measured, and the most common cause of the remaining
  wrong answers.
- **Small percentages:** whole percentages up to 31% are still skipped by the day-of-month rule.
