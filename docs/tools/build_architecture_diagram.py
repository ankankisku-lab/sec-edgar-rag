"""Build docs/figures/architecture_overview.png: the full architecture on one page, with the key
numbers and comparisons for every pipeline stage.

Result numbers come from report_data.py (the same evidence layer as the report); prose facts
(README measurements, design notes) are written in the card text, as in build_report.py.
Run:  .venv\\Scripts\\python docs\\tools\\build_architecture_diagram.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import PercentFormatter
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import report_data as R  # noqa: E402

OUT = R.ROOT / "docs" / "figures" / "architecture_overview.png"
V0, V1 = "retrieval_v0.jsonl", "retrieval_v1.jsonl"
W, H = 20.0, 33.0
INK, MUTED, GRID = "#1F2937", "#5B6472", "#D5DAE1"
C_OFF, C_STORE, C_RET = "#2F5496", "#17736B", "#6B4C9A"
C_LLM, C_VER, C_EVAL = "#B8561B", "#3F7D3A", "#3B4A5E"
C_BAD, C_GOOD = "#B42318", "#2E7D32"
GREY, LIGHT = "#9AA5B4", "#D7DCE3"
fig = ax = None


def f3(x):
    return f"{x:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def sig_txt(s, metric=""):
    return f"{s['diff']:+.3f}{' ' + metric if metric else ''}, {'sig.' if s['sig'] else 'n.s.'}"


# ====================================================================== drawing helpers
def tint(hexc, a):
    h = hexc.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * a) for c in (r, g, b))


def rbox(x, y, w, h, fc, ec, lw=1.4, r=0.08, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def lines(x, y, items, width_in, fs=9.2, color=INK, lh=1.42):
    """Bullet lines with hanging indent. A trailing '|style' sets b(old), x (red), k (green), n (no bullet)."""
    step = fs * lh / 72
    chars = max(20, int(width_in / (0.54 * fs / 72)))
    for text in items:
        style = ""
        if "|" in text and set(text.rsplit("|", 1)[1]) <= set("bxkn"):
            text, style = text.rsplit("|", 1)
        col = C_BAD if "x" in style else C_GOOD if "k" in style else color
        bullet = "" if "n" in style else "•  "
        for i, ln in enumerate(textwrap.wrap(text, chars - len(bullet)) or [""]):
            prefix = bullet if i == 0 else " " * (len(bullet) + 1 if bullet else 0)
            ax.text(x, y, prefix + ln, fontsize=fs, color=col, fontweight="bold" if "b" in style else "normal", va="top", zorder=5)
            y -= step
        y -= step * 0.18
    return y


def hbars(x, y, w, rows, fs=8.4, label_w=1.35, vmax=1.0, fmt="{:.3f}"):
    """rows: (label, value, color). Returns the next y."""
    bh, step = 0.16, 0.25
    bx0, bx1 = x + label_w, x + w - 0.55
    for label, v, col in rows:
        ax.text(x, y - bh / 2, label, fontsize=fs, va="center", color=INK, zorder=5)
        ax.add_patch(Rectangle((bx0, y - bh), bx1 - bx0, bh, fc="white", ec=GRID, lw=0.6, zorder=4))
        ax.add_patch(Rectangle((bx0, y - bh), (bx1 - bx0) * max(v, 0.004) / vmax, bh, fc=col, ec="none", zorder=5))
        ax.text(bx1 + 0.06, y - bh / 2, fmt.format(v), fontsize=fs, va="center", color=INK, fontweight="bold", zorder=5)
        y -= step
    return y - 0.04


def title_bar(x, top, w, color, title, path=None, bar=0.38, fs=12):
    ax.add_patch(FancyBboxPatch((x, top - bar), w, bar, boxstyle="round,pad=0,rounding_size=0.08", fc=color, ec=color, zorder=2))
    ax.add_patch(Rectangle((x, top - bar), w, bar / 2, fc=color, ec=color, zorder=2))
    ax.text(x + 0.15, top - bar / 2, title, color="white", fontsize=fs, fontweight="bold", va="center", zorder=3)
    if path:
        ax.text(x + w - 0.15, top - bar / 2, path, color=tint(color, 0.75), fontsize=8.3, family="monospace", ha="right", va="center", zorder=3)
    return bar


def card(x, top, w, h, color, title, path, left, right_title, right=None, right_draw=None, split=0.56):
    """Pipeline node: description on the left, numbers / comparison panel on the right."""
    rbox(x, top - h, w, h, "white", color, lw=1.6)
    bar = title_bar(x, top, w, color, title, path)
    lines(x + 0.18, top - bar - 0.16, left, w * split - 0.32)
    rx, rw = x + w * split, w * (1 - split) - 0.12
    rtop, rh = top - bar - 0.1, h - bar - 0.2
    rbox(rx, rtop - rh, rw, rh, tint(color, 0.91), "none", r=0.06, z=2)
    ax.text(rx + 0.12, rtop - 0.1, right_title, fontsize=9.2, fontweight="bold", color=color, va="top", zorder=5)
    y = rtop - 0.42
    if right_draw:
        y = right_draw(rx + 0.12, y, rw - 0.24)
    if right:
        lines(rx + 0.12, y, right, rw - 0.24, fs=8.7)


def pill(x, top, w, h, text, color):
    rbox(x, top - h, w, h, tint(color, 0.85), color, lw=1.6, r=0.2)
    ax.text(x + w / 2, top - h / 2, text, fontsize=10.5, ha="center", va="center", color=INK, fontweight="bold", zorder=5)


def down_arrow(x, y1, y2, color="#555"):
    ax.annotate("", xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16), zorder=6)


def section(x, y, text, sub=None, color=INK):
    ax.text(x, y, text, fontsize=16, fontweight="bold", color=color, va="bottom")
    if sub:
        ax.text(x, y - 0.08, sub, fontsize=9.5, color=MUTED, va="top")


def panel(x, top, w, h, title, color=C_EVAL):
    rbox(x, top - h, w, h, "white", color, lw=1.4)
    bar = title_bar(x, top, w, color, title, bar=0.36, fs=11)
    return x + 0.1, top - h + 0.1, w - 0.2, h - bar - 0.2


def axes_at(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GREY)
    a.tick_params(labelsize=8.5, colors=INK)
    return a


def table(x, y, cols, header, rows, fs=8.3, bold_rows=()):
    for c, h in zip(cols, header):
        ax.text(x + c, y, h, fontsize=fs - 0.3, fontweight="bold", color=MUTED, va="top")
    y -= 0.26
    ax.plot([x, x + cols[-1] + 0.55], [y + 0.05, y + 0.05], color=GRID, lw=0.8)
    for i, r in enumerate(rows):
        for c, v in zip(cols, r):
            ax.text(x + c, y, v, fontsize=fs, va="top", color=INK, fontweight="bold" if i in bold_rows else "normal")
        y -= 0.235
    return y


def link(y_from, y_to, x_from, x_gap, x_to, label, color):
    ax.plot([x_from, x_gap, x_gap], [y_from, y_from, y_to], color=color, lw=2.2, zorder=7, solid_capstyle="round")
    ax.annotate("", xy=(x_to, y_to), xytext=(x_gap, y_to), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16), zorder=7)
    ax.text(x_gap - 0.07, (y_from + y_to) / 2, label, rotation=90, fontsize=8.2, color=color, fontweight="bold", ha="right",
            va="center", bbox=dict(fc="white", ec="none", pad=1.5), zorder=8)


# ====================================================================== build
def build(out: Path = OUT) -> Path:
    global fig, ax
    c, es, exp, sc, sw, cfg = R.corpus(), R.eval_sets(), R.experiments(), R.scaling(), R.sweeps(), R.configs()
    G = {g: R.gen_summary(g) for g in ("G1", "G2", "G3", "G4", "G5", "G6")}
    mc, ja, feas = R.mcnemar("G3", "G4"), R.judge_agreement(), R.feasibility()
    e0, e1 = exp[exp.dataset == V0].set_index("version"), exp[exp.dataset == V1].set_index("version")
    B = R.bootstrap
    sig = {"v1_v2": B("V1_dense", "V2_bm25", "mrr@10", V0), "v2_v3": B("V2_bm25", "V3_hybrid", "mrr@10", V0),
           "md_v4": B("V4-md_rerank", "V4_rerank", "mrr@10", V0), "v4_v4h": B("V4_rerank", "V4h_rerank", "mrr@10", V0),
           "v4_v5": B("V4_rerank", "V5_decomposed", "all_hit@10", V1), "v5_v6": B("V5_decomposed", "V6_decomposed", "mrr@10", V1),
           "s_v5": B("V5@S24_decomposed", "V5@Sall_decomposed", "mrr@10", V1), "s_v4": B("V4@S24_rerank", "V4@Sall_rerank", "mrr@10", V1)}
    fs_, cp = sw["fusion_sweep"], sw["candidate_pools"].set_index("pool")
    fus_rel = fs_[(fs_.fusion == "relative") & (fs_.alpha.astype(str) == "0.3") & (fs_.split == "dev")]["mrr@10"].iloc[0]
    fus_rrf = fs_[(fs_.fusion == "rrf") & (fs_.split == "dev")]["mrr@10"].iloc[0]
    first, last = sc.iloc[0], sc.iloc[-1]
    parents = c["parents_text"] + c["parents_table"]
    max_unverified = max(g["unverified"] for g in G.values())
    bm25, rr = cfg["retrieval"]["bm25"], cfg["retrieval"]["rerank"]
    budget, ctx_win = cfg["generation"]["context"]["budget_tokens"], cfg["generation"]["llm"]["context_window"]

    fig = plt.figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ------------------------------------------------------------ header + KPI tiles
    ax.text(0.4, 32.55, "SEC-EDGAR Financial RAG & Evaluation Engine — full architecture", fontsize=27, fontweight="bold", color=INK, va="center")
    ax.text(0.4, 32.08, "Retrieval-augmented question answering over Nasdaq-100 10-K / 10-Q filings, built phase by phase on a 4 GB-VRAM "
            "laptop with free models only. Every component was kept only if it beat a measured baseline.", fontsize=11.5, color=MUTED, va="center")
    ax.text(0.4, 31.78, f"Numbers from docs/tools/report_data.py at commit {R.head_commit()} (same evidence as the architecture report).  "
            "✓ adopted  ✗ tried and rejected  sig. = paired-bootstrap 95% CI excludes 0 · n.s. = not significant",
            fontsize=9.5, color=MUTED, va="center", style="italic")
    kpis = [(str(c["n_filings"]), "filings", f"{c['forms'].get('10-K', 0)} 10-K + {c['forms'].get('10-Q', 0)} 10-Q · {c['n_filers_with_filings']} companies"),
            (f"{c['children']:,}", "child chunks", f"+ {parents:,} parent sections"),
            (f"{f3(e1.loc['V1', 'mrr@10'])} → {f3(e1.loc['V5', 'mrr@10'])}", "retrieval MRR@10", "dense V1 → final V5 · eval v1"),
            (f"{f3(e1.loc['V4', 'all_hit@10'])} → {f3(e1.loc['V5', 'all_hit@10'])}", "all-hit@10", "every fact in top 10 · V4 → V5"),
            (f"{pct(G['G3']['correct'])} → {pct(G['G4']['correct'])}", "answer accuracy", f"G3 → G4 · McNemar p = {mc['p']:.4f}"),
            (pct(max_unverified, 0), "unverified numbers", "answers with an unsupported number, G1–G6"),
            (f"{last['V5_latency_p50_ms']:.0f} ms", "retrieval p50", f"V5 at the full {int(last.filings)}-filing index"),
            ("4 GB", "VRAM budget", "RTX 3050 laptop · 16 GB RAM · free models")]
    kw, kg, ktop, kh = (19.2 - 7 * 0.18) / 8, 0.18, 31.45, 1.12
    for i, (big, lab, sub) in enumerate(kpis):
        x = 0.4 + i * (kw + kg)
        rbox(x, ktop - kh, kw, kh, tint(C_OFF, 0.93), tint(C_OFF, 0.6), lw=1.0)
        ax.text(x + kw / 2, ktop - 0.3, big, fontsize=15 if len(big) < 10 else 12.5, fontweight="bold", color=C_OFF, ha="center", va="center")
        ax.text(x + kw / 2, ktop - 0.63, lab, fontsize=9.5, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(x + kw / 2, ktop - 0.88, sub, fontsize=7.4, color=MUTED, ha="center", va="center")

    LX, RX, CW = 0.4, 10.3, 9.3
    TOP, BOT, GAP = 29.55, 12.35, 0.3
    section(LX, 29.8, "A · OFFLINE — ingestion, parsing, indexing", color=C_OFF)
    section(RX, 29.8, "B · ONLINE — query, retrieval, grounded answer", color=C_RET)

    # ------------------------------------------------------------ A: offline cards
    def dense_vs_bm25(x, y, w):
        y = hbars(x, y, w, [("P@3 numeric · dense", e0.loc["V1", "numeric_precision@3"], GREY),
                            ("P@3 numeric · BM25 ✓", e0.loc["V2", "numeric_precision@3"], C_OFF)], vmax=0.6, label_w=1.75)
        y = hbars(x, y + 0.02, w, [("MRR@10 · dense", e0.loc["V1", "mrr@10"], GREY),
                                   ("MRR@10 · BM25 ✓", e0.loc["V2", "mrr@10"], C_OFF)], vmax=0.6, label_w=1.75)
        lines(x, y + 0.06, [f"BM25: {sig_txt(sig['v1_v2'], 'MRR')}"], w, fs=8.5)
        return y

    n_sections = c["section_mode"]
    off = [
        (C_OFF, "1 · Corpus manifest", "src/ingestion/metadata.py",
         ["Wikipedia Nasdaq-100 list → SEC ticker/CIK map → SEC submissions API",
          "Keeps the latest 2 × 10-K + 6 × 10-Q per filer",
          f"Share classes merged (GOOG/GOOGL) → {c['n_companies_listed']} filers",
          f"{len(c['foreign'])} foreign private issuers file 20-F/40-F → excluded by design ({', '.join(c['foreign'])})",
          "Output: data/metadata/filings.csv, companies.csv"],
         "Corpus", [f"{c['n_filings']} filings from {c['n_filers_with_filings']} companies|b",
                    f"{c['forms'].get('10-K', 0)} annual 10-K · {c['forms'].get('10-Q', 0)} quarterly 10-Q",
                    f"filed {c['filing_dates'][0]} → {c['filing_dates'][1]}",
                    "Scaled gradually: " + " → ".join(str(int(f)) for f in sc.filings) + " filings"], None),
        (C_OFF, "2 · Resumable download", "src/ingestion/downloader.py",
         ["Rate-limited SEC client: 5 req/s (SEC allows 10)", "Validated, atomic writes + a metadata sidecar per file",
          "JSON checkpoints: a crash resumes where it stopped", "Output: data/raw/<TICKER>/…htm"],
         "Numbers & fixes", [f"{c['raw_gb']:.2f} GB of primary documents|b",
                             "✗ Windows file locks refused the checkpoint rename|x", "✓ fixed with short retries of os.replace|k"], None),
        (C_OFF, "3 · iXBRL verification", "src/ingestion/verify.py",
         ["Reads each filing's own cover tags: dei:DocumentType, DocumentPeriodEndDate, EntityCentralIndexKey",
          "Compares them with the manifest (form, period end, CIK)",
          "Records fiscal year + period → period labels (NVIDIA's quarter ending Jul 2026 = fiscal 2027 Q2)"],
         "Result", [f"{c['verified_ok']} / {c['verified_total']} filings match|b", "✗ nested ix:nonNumeric tags gave false mismatches|x",
                    "✓ nesting-aware extraction|k", "Output: data/metadata/verification.csv"], None),
        (C_OFF, "4 · Structure-preserving parser (v4)", "src/parsing/",
         ["DOM → text blocks, tables, page breaks; running headers, footers and page numbers removed",
          "Part / Item / Note headings tracked with ordering guards",
          "Tables rebuilt: colspan expansion, '$' and ')' cells merged, header detection, continuation tables inherit headers; captions, units, footnotes",
          "Output: data/processed/<TICKER>/<TICKER>_<FORM>_<PERIOD>.json"],
         "Numbers & comparison", [f"{c['n_tables']:,} tables · {c['n_footnotes']:,} linked footnotes|b",
                                  f"{n_sections.get('items', 0)} standard Items · {n_sections.get('titles', 0)} title fallback (INTC, HON 10-Ks)",
                                  "✗ default HTML reader: tables lose structure|x", "v4: year rows + period labels as headers (MSFT)"], None),
        (C_OFF, "5 · Parent–child chunking (v3)", "src/chunking/hierarchical.py",
         ["LlamaIndex TextNodes with PARENT / CHILD / PREV / NEXT links",
          f"Parents: heading groups ≤ {cfg['chunking']['text']['parent_max_tokens']:,} tok · whole tables ≤ {cfg['chunking']['tables']['parent_max_tokens']:,} tok",
          f"Children: ~{cfg['chunking']['text']['child_tokens']}-tok sentence splits · table row groups that repeat the period header",
          "Metadata header prepended: company, ticker, form, fiscal period, heading path, caption, units",
          "IDs uuid5(accession/p{i}/c{j}) → re-runs never duplicate", "Output: data/chunks/<TICKER>/<filing>.jsonl"],
         "Numbers & comparison", [f"{c['children']:,} children · {parents:,} parents|b",
                                  f"{c['parents_text']:,} text + {c['parents_table']:,} table parents",
                                  f"Child tokens: median {c['child_tok_p50']:.0f}, max {c['child_tok_max']} (< 512 limit)",
                                  "✗ HierarchicalNodeParser: not structure-aware|x"], None),
        (C_OFF, "6 · Encode: dense + sparse", "src/embeddings/ · src/retrieval/bm25.py",
         [f"Dense: bge-small-en-v1.5 bi-encoder, {cfg['embedding']['dim']}-d, fp16 on GPU, batch {cfg['embedding']['batch_size']}, 298 MiB peak, per-filing vector cache",
          f"Sparse: BM25 (fastembed Qdrant/bm25), k = {bm25['k']}, b = {bm25['b']}, avg_len = {bm25['avg_len']} (measured; default 256)",
          "IDF applied by Qdrant at query time → stays correct as filings are added",
          "Dense weakness: one vector per 15-row table dilutes the single row asked for"],
         "Dense vs BM25 (eval v0)", None, dense_vs_bm25),
        (C_STORE, "7 · Stores", "qdrant_index.py · parent_store.py",
         ["Qdrant v1.19.1 (Docker, named volume) holds the children: text-dense (384-d cosine, HNSW) + text-sparse-new (IDF, inverted index) + payload indexes (ticker, form, fiscal period, item, dates)",
          "SQLite (WAL) parent store: parents are only fetched by id, never searched → Qdrant RAM stays for children",
          "Idempotent: a filing's points are deleted by accession before re-upsert"],
         f"Footprint at {int(last.filings)} filings", [f"Qdrant RAM {last.qdrant_mem} · disk {last.qdrant_disk}|b", f"Parents DB {last.parents_db_mb:.0f} MB",
                                                     "127.0.0.1 vs 'localhost': ~8 ms vs ~2 s per request → indexing 24 filings 3 m 46 s → ~21 s",
                                                     "Top-10 identical to brute-force search"], None),
    ]
    off_h = (TOP - BOT - 6 * GAP) / 7
    off_tops, y = [], TOP
    for i, spec in enumerate(off):
        card(LX, y, CW, off_h, *spec)
        off_tops.append(y)
        if i < len(off) - 1:
            down_arrow(LX + CW * 0.28, y - off_h, y - off_h - GAP)
        y -= off_h + GAP

    # ------------------------------------------------------------ B: online cards
    def decomp_bars(x, y, w):
        bx0, bx1 = x + 1.2, x + w - 1.05
        for lab, key in [("all questions", "all_hit@10"), ("cross-quarter", "cross_quarter_all_hit@10"), ("cross-company", "cross_company_all_hit@10")]:
            a, b = e1.loc["V4", key], e1.loc["V5", key]
            ax.text(x, y - 0.1, lab, fontsize=8.4, va="center", color=INK, zorder=5)
            for k, (v, col) in enumerate([(a, GREY), (b, C_RET)]):
                yy = y - 0.01 - k * 0.11
                ax.add_patch(Rectangle((bx0, yy - 0.09), bx1 - bx0, 0.09, fc="white", ec=GRID, lw=0.5, zorder=4))
                ax.add_patch(Rectangle((bx0, yy - 0.09), (bx1 - bx0) * max(v, 0.004), 0.09, fc=col, ec="none", zorder=5))
            ax.text(bx1 + 0.06, y - 0.1, f"{f3(a)} → {f3(b)}", fontsize=8.4, va="center", fontweight="bold", color=INK, zorder=5)
            y -= 0.31
        return y - 0.02

    def retr_bars(x, y, w):
        return hbars(x, y, w, [("dense V1", e1.loc["V1", "mrr@10"], GREY), ("BM25 V2 ✓", e1.loc["V2", "mrr@10"], C_RET),
                               ("hybrid V3 ✗", e1.loc["V3", "mrr@10"], "#C9B8E0")], vmax=0.8)

    def rerank_bars(x, y, w):
        return hbars(x, y, w, [("BM25 only V2", e0.loc["V2", "mrr@10"], GREY), ("markdown V4-md ✗", e0.loc["V4-md", "mrr@10"], "#C9B8E0"),
                               ("linearized V4 ✓", e0.loc["V4", "mrr@10"], C_RET)], label_w=1.6)

    def ctx_bars(x, y, w):
        return hbars(x, y, w, [("G3 · V4 retrieval", G["G3"]["ctx"], GREY), ("G4 · V5 retrieval", G["G4"]["ctx"], C_RET)],
                     fmt="{:.1%}", label_w=1.6)

    def llm_table(x, y, w):
        f = feas.assign(both=lambda d: d.has_2025_value & d.has_2024_value,
                        unsupported=lambda d: d.unsupported_numbers.fillna("").astype(str).str.len() > 0)
        summ = f.groupby("model").agg(runs=("model", "size"), correct=("both", "sum"), wrong=("unsupported", "sum"))
        speed = f[f.kv_cache == "q8_0"].pivot(index="model", columns="context", values="gen_tok_s")
        best = summ.sort_values(["correct", "wrong"], ascending=[False, True]).index[0]
        cols = [0, 1.45, 2.2, 2.95]
        for col, hd in zip(cols, ["model (Q4_K_M)", "tok/s", "correct", "unsupported"]):
            ax.text(x + col, y, hd, fontsize=8.2, color=MUTED, va="top", fontweight="bold")
        y -= 0.25
        for m in speed[4096].sort_values(ascending=False).index:
            good = m == best
            if good:
                ax.add_patch(Rectangle((x - 0.06, y - 0.2), w + 0.1, 0.24, fc=tint(C_LLM, 0.75), ec="none", zorder=4))
            row = (m + (" ✓" if good else ""), f"{speed.loc[m, 4096]:.1f}", f"{int(summ.loc[m, 'correct'])}/{int(summ.loc[m, 'runs'])}",
                   f"{int(summ.loc[m, 'wrong'])}/{int(summ.loc[m, 'runs'])}")
            for col, v in zip(cols, row):
                ax.text(x + col, y, v, fontsize=8.6, va="top", color=INK, fontweight="bold" if good else "normal", zorder=5)
            y -= 0.24
        return y - 0.06

    def acc_bars(x, y, w):
        return hbars(x, y, w, [("G2 · prompt v1", G["G2"]["correct"], GREY), ("G3 · prompt v2", G["G3"]["correct"], tint(C_VER, 0.45)),
                               ("G4 · + decomposition", G["G4"]["correct"], C_VER)], fmt="{:.1%}", label_w=1.75)

    on = [
        (C_RET, "8 · Router + sub-question decomposition", "src/query/decomposition.py",
         ["Regex router: only comparison / multi-period questions reach the LLM; the rest keep the original question",
          "Qwen3-4B, JSON mode, one call: one lookup per company / metric / period",
          "One sub-question back → original wording kept ('What drove X' was becoming 'What was X')",
          "✗ SubQuestionQueryEngine: 4–5 LLM calls vs 2|x"],
         "all-hit@10: V4 (grey) → V5 (colour), eval v1",
         [f"Largest single gain: {sig_txt(sig['v4_v5'])}|b", f"Router: p50 retrieval 2.3 s → {e1.loc['V5', 'latency_p50_ms'] / 1000:.1f} s"], decomp_bars),
        (C_RET, f"9 · Candidate retrieval: BM25 top-60", "Qdrant sparse · src/retrieval/bm25.py",
         ["Each sub-question searched on the sparse (BM25) vectors, 60 candidates",
          "Questions name exact line items ('operating income') and periods — lexical match wins",
          f"Hybrid tuned on the dev split: relative-score α = 0.3 (dev MRR {fus_rel:.3f}) beat 0.5, 0.7 and RRF ({fus_rrf:.3f})",
          f"…yet hybrid V3 vs BM25 on v0: {sig_txt(sig['v2_v3'])}|x"],
         "MRR@10 by retriever (eval v1)",
         [f"Pool recall @60: BM25 {pct(cp.loc['bm25 top-60', 'hit'])} · dense {pct(cp.loc['dense top-60', 'hit'])}",
          f"p50: BM25 {e1.loc['V2', 'latency_p50_ms']:.0f} ms · dense {e1.loc['V1', 'latency_p50_ms']:.0f} ms · hybrid {e1.loc['V3', 'latency_p50_ms']:.0f} ms"], retr_bars),
        (C_RET, "10 · Cross-encoder rerank", "src/retrieval/reranker.py · linearize.py",
         ["bge-reranker-base (XLM-RoBERTa, 12 layers) reads question + candidate jointly → checks line item AND period",
          "Ranks on raw logits (sigmoid saturated near 0.999 → ties)",
          "Tables linearized: 'Operating income: Three Months Ended June 28, 2025 = 28,202; …'",
          f"fp16, batch {rr['batch_size']}: 439 ms / 906 MiB per 60 pairs (batch 32: 939 ms / 1,218 MiB)"],
         "MRR@10 on eval v0", [f"Linearization alone: {sig_txt(sig['md_v4'])}|b",
                              f"V2 → V4 on v1: {f3(e1.loc['V2', 'mrr@10'])} → {f3(e1.loc['V4', 'mrr@10'])}",
                              f"✗ + dense candidates (V4h): {sig_txt(sig['v4_v4h'])}|x"], rerank_bars),
        (C_RET, "11 · Interleave + parent expansion", "decomposed.py · generation/context.py",
         ["Round-robin interleave of the sub-question rankings → top 8, so every fact sits near the top",
          "Each child swapped for its parent section (SQLite)",
          f"Packed into a {budget:,}-token budget inside the {ctx_win // 1024}k LLM context",
          "Sources numbered and prefixed with the metadata header"],
         f"All facts in the LLM context ({G['G4']['n']} q)", [f"G1 (numeric only, V4): gold in context {pct(G['G1']['ctx'])}"], ctx_bars),
        (C_LLM, "12 · Grounded generation", "src/generation/generator.py",
         ["LlamaIndex CitationQueryEngine → Qwen3-4B-Instruct-2507 via Ollama",
          f"Q4_K_M · q8_0 KV cache · flash attn · {ctx_win // 1024}k ctx · temp 0",
          "~2.25 GB of layers on GPU, the rest on CPU",
          "Prompt v2: <calc> for every change, state differences, refuse only when no figure exists"],
         "LLM feasibility (4k ctx, all runs)", ["Chosen on correctness, not speed|b"], llm_table),
        (C_VER, "13 · <calc> + number verification", "src/generation/numbers.py",
         ["The LLM writes <calc> expressions; Python evaluates them (every benchmark error was a subtraction or a misread value)",
          "Every number in the answer is checked against the sources",
          "Unsupported numbers are listed with the answer, never hidden"],
         f"Answer accuracy (same {G['G4']['n']} questions)",
         [f"{pct(max_unverified, 0)} answers with unverified numbers (G1–G6)|b",
          f"G3 → G4: {mc['fixed']} fixed, {mc['broke']} broken, p = {mc['p']:.4f}"], acc_bars),
    ]
    PILL = 0.52
    on_h = (TOP - BOT - 2 * PILL - 7 * GAP) / 6
    y = TOP
    pill(RX, y, CW, PILL, "User question — often multi-company or multi-period", C_RET)
    down_arrow(RX + CW * 0.28, y - PILL, y - PILL - GAP); y -= PILL + GAP
    on_tops = []
    for spec in on:
        card(RX, y, CW, on_h, *spec)
        on_tops.append(y)
        down_arrow(RX + CW * 0.28, y - on_h, y - on_h - GAP)
        y -= on_h + GAP
    pill(RX, y, CW, PILL, f"Answer with [n] citations · sources · calculations · unverified numbers  ·  p50 {G['G4']['p50_s']:.1f} s (G4)", C_VER)

    # stores feed the online pipeline
    store_top = off_tops[-1]
    link(store_top - off_h * 0.35, on_tops[1] - on_h * 0.5, LX + CW, 9.93, RX, "children: dense + sparse + payload", C_STORE)
    link(store_top - off_h * 0.62, on_tops[3] - on_h * 0.5, LX + CW, 10.15, RX, "parents by id", C_STORE)

    # ------------------------------------------------------------ C: evaluation
    t1 = es[V1]["types"]
    section(LX, 11.85, "C · EVALUATION — how every component was measured", color=C_EVAL,
            sub=f"Deterministic gold from statement rows (period validated) · eval v0: {es[V0]['n']} questions · eval v1: {es[V1]['n']} questions "
                f"in {len(t1)} types with one gold group per fact · settings chosen on a dev split · paired bootstrap (10,000 resamples) and McNemar")
    R1T, R1H = 11.05, 4.0
    pw = (19.2 - 2 * 0.3) / 3

    # MRR ladder
    ix, iy, iw, ih = panel(LX, R1T, pw, R1H, "Retrieval MRR@10 by version (eval v1, p50 latency below)")
    a = axes_at(ix + 0.45, iy + 0.62, iw - 0.55, ih - 0.75)
    ladder = [("V1", "dense", GREY), ("V1h", "HyDE", LIGHT), ("V2", "BM25", C_EVAL), ("V3", "hybrid", LIGHT),
              ("V4", "+rerank", C_EVAL), ("V5", "+decomp", C_EVAL), ("V5h", "ctrl", LIGHT), ("V6", "+HyDE", LIGHT)]
    vals = [e1.loc[v, "mrr@10"] for v, _, _ in ladder]
    a.bar(range(len(ladder)), vals, color=[col for *_, col in ladder], width=0.7)
    for i, (v, (_, _, col)) in enumerate(zip(vals, ladder)):
        a.text(i, v + 0.015, f3(v), ha="center", fontsize=8.3, fontweight="bold" if col == C_EVAL else "normal")
    lat = [e1.loc[v, "latency_p50_ms"] for v, _, _ in ladder]
    a.set_xticks(range(len(ladder)))
    a.set_xticklabels([f"{v}\n{n}\n{ms / 1000:.1f} s" if ms >= 1000 else f"{v}\n{n}\n{ms:.0f} ms" for (v, n, _), ms in zip(ladder, lat)], fontsize=7.8)
    a.set_ylim(0, 0.85); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
    a.text(0.01, 0.97, "dark = adopted path   light = rejected", transform=a.transAxes, fontsize=8, color=MUTED, va="top")

    # all-hit by question type
    ix, iy, iw, ih = panel(LX + pw + 0.3, R1T, pw, R1H, "Every fact retrieved in top 10 (all-hit@10) by question type")
    a = axes_at(ix + 0.45, iy + 0.45, iw - 0.55, ih - 0.62)
    types = ["numeric", "yoy", "pct_change", "cross_quarter", "cross_company", "narrative"]
    for k, (v, lab, col) in enumerate([("V2", "V2 BM25", "#C3CAD4"), ("V4", "V4 +rerank", "#7C8BA1"), ("V5", "V5 +decomp", C_EVAL)]):
        vs = [e1.loc[v, f"{t}_all_hit@10"] for t in types]
        a.bar([i + (k - 1) * 0.26 for i in range(len(types))], vs, 0.24, label=lab, color=col)
        for i, val in enumerate(vs):
            if val == 0:
                a.text(i + (k - 1) * 0.26, 0.02, "0", ha="center", fontsize=8.5, color=C_BAD, fontweight="bold")
    a.set_xticks(range(len(types))); a.set_xticklabels([t.replace("cross_", "cross_\n") for t in types], fontsize=8)
    a.set_ylim(0, 1.18); a.set_yticks([0, 0.5, 1.0])
    a.legend(fontsize=8, ncol=3, frameon=False, loc="upper left"); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
    a.text(3.5, 0.3, "0 = no fact\nretrieved", ha="center", fontsize=7.8, color=C_BAD, bbox=dict(fc="white", ec=C_BAD, lw=0.6, pad=2))

    # answer accuracy by type
    ix, iy, iw, ih = panel(LX + 2 * (pw + 0.3), R1T, pw, R1H, "Answer accuracy by type: G3 (no decomposition) vs G4")
    a = axes_at(ix + 0.45, iy + 0.45, iw - 0.55, ih - 0.62)
    gt = [t for t in types if t in G["G4"]["by_type"]]
    g3, g4 = [G["G3"]["by_type"].get(t, 0) for t in gt], [G["G4"]["by_type"].get(t, 0) for t in gt]
    a.bar([i - 0.19 for i in range(len(gt))], g3, 0.38, color="#C3CAD4", label=f"G3  {pct(G['G3']['correct'])} overall")
    a.bar([i + 0.19 for i in range(len(gt))], g4, 0.38, color=C_VER, label=f"G4  {pct(G['G4']['correct'])} overall")
    for i, (p, q) in enumerate(zip(g3, g4)):
        a.text(i - 0.19, p + 0.02, f"{p:.0%}", ha="center", fontsize=7.5)
        a.text(i + 0.19, q + 0.02, f"{q:.0%}", ha="center", fontsize=7.5, fontweight="bold")
    a.set_xticks(range(len(gt))); a.set_xticklabels([t.replace("cross_", "cross_\n") for t in gt], fontsize=8)
    a.set_ylim(0, 1.25); a.set_yticks([0, 0.5, 1.0]); a.yaxis.set_major_formatter(PercentFormatter(1.0))
    a.legend(fontsize=8, ncol=2, frameon=False, loc="upper left"); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)

    R2T, R2H = 6.75, 3.5
    qw = (19.2 - 3 * 0.3) / 4
    # scaling
    ix, iy, iw, ih = panel(LX, R2T, qw, R2H, "Scaling: quality vs corpus size")
    a = axes_at(ix + 0.45, iy + 1.3, iw - 0.55, ih - 1.4)
    for v, col in [("V2", "#C3CAD4"), ("V4", "#7C8BA1"), ("V5", C_EVAL)]:
        a.plot(sc.filings, sc[f"{v}_mrr@10"], marker="o", color=col, label=v, lw=2, ms=4)
    a.set_xscale("log"); a.set_xticks(sc.filings); a.set_xticklabels([str(int(f)) for f in sc.filings], fontsize=8); a.minorticks_off()
    a.set_ylim(0.35, 0.8); a.set_ylabel("MRR@10", fontsize=8.5); a.set_xlabel("filings in index (log scale)", fontsize=8.3, labelpad=2)
    a.legend(fontsize=8, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02)); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    lines(ix + 0.05, iy + 0.66, [f"Chunks {int(first.points):,} → {int(last.points):,} · p50 {first['V5_latency_p50_ms']:.0f} → {last['V5_latency_p50_ms']:.0f} ms",
                                f"V5 MRR {sig_txt(sig['s_v5'])} · V4 {sig_txt(sig['s_v4'])}",
                                f"Qdrant RAM {first.qdrant_mem} → {last.qdrant_mem}"], iw - 0.1, fs=8.3, lh=1.3)

    # generation ledger
    x0 = LX + qw + 0.3
    ix, iy, iw, ih = panel(x0, R2T, qw, R2H, "Generation runs (answer-keyed questions)")
    yy = table(ix + 0.05, iy + ih - 0.05, [0, 0.55, 1.05, 1.85, 2.75, 3.6], ["run", "n", "correct", "facts in ctx", "refused", "p50 s"],
               [(g, str(s["n"]), pct(s["correct"]), pct(s["ctx"]), pct(s["refused"]), f"{s['p50_s']:.1f}") for g, s in G.items()],
               bold_rows=(3,))
    lines(ix + 0.05, yy - 0.08, ["G1 numeric, V4 · G2 prompt v1 · G3 prompt v2 · G4 = G3 + decomposition",
                                 "G5 / G6 reproduce G3 / G4 exactly → deterministic"
                                 if (G["G5"]["correct"], G["G6"]["correct"]) == (G["G3"]["correct"], G["G4"]["correct"])
                                 else "G5 / G6 re-run G3 / G4 with contexts saved for Ragas",
                                 f"Unverified-number answers: {pct(max_unverified, 0)} in every run|b"], iw - 0.1, fs=8.1, lh=1.3)

    # judges
    x0 += qw + 0.3
    ix, iy, iw, ih = panel(x0, R2T, qw, R2H, "Ragas judges (Phase 13, paused)")
    yy = table(ix + 0.05, iy + ih - 0.05, [0, 1.3, 1.95, 2.55, 3.35], ["metric", "120B", "8B", "Pearson", "same side"],
               [(m.replace("answer_", ""), f3(ja[m]["a"]), f3(ja[m]["b"]), f"{ja[m]['pearson']:.2f}", pct(ja[m]["same_side"], 0))
                for m in ("faithfulness", "answer_relevancy")])
    lines(ix + 0.05, yy - 0.1, ["Judges: gpt-oss-120b on Groq free plan (8k tokens/min, ~200k/day) · Llama 3.1 8B local, schema-constrained",
                                f"Same {ja['n']} G5 answers judged by both ({ja['n'] + ja['refused']} scored, {ja['refused']} refusals excluded)",
                                "✓ 8B valid for relevancy|k",
                                f"✗ 8B not reliable for faithfulness: its {len(ja['flips'])} disagreements were correct answers|x",
                                "Paused at the free-tier daily limit; resumes from cache"], iw - 0.1, fs=8.1, lh=1.3)

    # rejected
    x0 += qw + 0.3
    ix, iy, iw, ih = panel(x0, R2T, qw, R2H, "Tried, measured, rejected", color=C_BAD)
    lines(ix + 0.05, iy + ih - 0.05, [
        f"✗ Dense retrieval (V1): numeric P@3 {f3(e0.loc['V1', 'numeric_precision@3'])}; right company, wrong row|x",
        f"✗ Hybrid fusion (V3) vs BM25 on v0: {sig_txt(sig['v2_v3'], 'MRR')}|x",
        f"✗ RRF: dev MRR {fus_rrf:.3f} vs {fus_rel:.3f} for relative-score α 0.3|x",
        f"✗ Dense candidates in the rerank pool (V4h): {sig_txt(sig['v4_v4h'])}|x",
        f"✗ Markdown tables for the reranker: {-sig['md_v4']['diff']:+.3f} vs linearized|x",
        f"✗ HyDE (V6 vs V5): {sig_txt(sig['v5_v6'])}, ~3 s extra per query|x",
        f"✗ Local 8B as faithfulness judge: Pearson {ja['faithfulness']['pearson']:.2f}|x",
        "✗ SubQuestionQueryEngine: 4–5 LLM calls per question|x",
        "Why: questions name exact line items and periods; once the cross-encoder reads each candidate, extra dense candidates rarely change the ranking|n",
    ], iw - 0.1, fs=8.1, lh=1.3)

    # ------------------------------------------------------------ footer
    FT, FH = 2.95, 1.75
    ix, iy, iw, ih = panel(LX, FT, 7.3, FH, "Hardware envelope", color=INK)
    lines(ix + 0.05, iy + ih - 0.05, [
        "RTX 3050 Laptop GPU, 4 GB VRAM · 16 GB RAM · Windows 11 · all data and model caches on D:",
        "Windows reserves ~0.8 GB and llama.cpp keeps ~1 GB free → the LLM gets ~2.25 GB; remaining layers run on CPU",
        "Embedder + reranker resident: 927 MiB · with Qwen3-4B loaded: 3.3 of 4.0 GB used",
        "✗ Orphaned llama-server runners held VRAM → prompt processing ~70 tok/s|x",
        "✓ stop() kills them; reranker warms up before the LLM loads|k",
    ], iw - 0.1, fs=8.3, lh=1.3)

    rbox(LX + 7.6, FT - FH, 6.2, FH, "white", MUTED, lw=1.4, ls=(0, (4, 3)))
    ax.text(LX + 7.75, FT - 0.2, "Serving layer (Phases 19–21, after the report)", fontsize=11, fontweight="bold", color=INK, va="center")
    lines(LX + 7.75, FT - 0.5, [
        "Phase 19: tracing with Arize Phoenix (local, open source)",
        "Phase 20: FastAPI service — query, health, metrics",
        "Chat UI served by the API at GET /",
        "Phase 21: Docker — full stack with GPU API, Ollama and Qdrant",
        "Not covered by the architecture report; summarised from the commit history|n",
    ], 5.9, fs=8.3, lh=1.3)

    ix, iy, iw, ih = panel(LX + 14.1, FT, 5.1, FH, "Legend", color=INK)
    for n, (col, lab) in enumerate([(C_OFF, "Offline ingestion"), (C_STORE, "Data stores"), (C_RET, "Online retrieval"),
                                    (C_LLM, "Local LLM"), (C_VER, "Verification / answer"), (C_EVAL, "Evaluation")]):
        lx, ly = ix + 0.05 + (n // 3) * 2.45, iy + ih - 0.1 - (n % 3) * 0.34
        ax.add_patch(Rectangle((lx, ly - 0.16), 0.3, 0.16, fc=col, ec="none"))
        ax.text(lx + 0.4, ly - 0.08, lab, fontsize=8.5, va="center", color=INK)
    ax.text(W - 0.4, FT - FH - 0.2, "Built by docs/tools/build_architecture_diagram.py from docs/tools/report_data.py",
            fontsize=8, color=MUTED, ha="right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    # trim the unused canvas below the footer
    im = Image.open(out).convert("RGB")
    l, t, r, b = ImageOps.invert(im).getbbox()
    m = 40
    im.crop((max(0, l - m), max(0, t - m), min(im.width, r + m), min(im.height, b + m))).save(out, optimize=True)
    return out


if __name__ == "__main__":
    print(build())
