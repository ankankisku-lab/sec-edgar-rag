"""Build docs/SEC_EDGAR_RAG_Architecture_Report.docx from the repository's code and results.

All numbers come from report_data.py, which reads result files; prose frames them.
Run:  .venv\\Scripts\\python docs\\tools\\build_report.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).parent))
import report_data as R  # noqa: E402

ROOT = R.ROOT
OUT = ROOT / "docs" / "SEC_EDGAR_RAG_Architecture_Report.docx"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
V0, V1 = "retrieval_v0.jsonl", "retrieval_v1.jsonl"
ACCENT = "#2F5496"


def f3(x):
    return f"{x:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ====================================================================== figures
def box(ax, x, y, w, h, text, fc="#DCE6F2", ec=ACCENT, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text.replace("->", "→"), ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))


def fig_offline(c):
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis("off")
    steps = [
        ("Wikipedia Nasdaq-100 list -> SEC ticker/CIK map\n-> SEC submissions API", "src/ingestion/metadata.py"),
        (f"Manifest: {c['n_filings']} filings (latest 2x 10-K + 6x 10-Q per filer)", "data/metadata/filings.csv"),
        ("Rate-limited, resumable download (checkpoint, validation)", "src/ingestion/downloader.py"),
        (f"iXBRL cover-tag verification ({c['verified_ok']}/{c['verified_total']} match)", "src/ingestion/verify.py"),
        ("Parse: blocks -> sections (Part/Item/Note) -> clean tables\n+ captions/units/footnotes; page artifacts removed", "src/parsing/"),
        ("Parent-child chunking (LlamaIndex TextNodes): text groups,\ntables with repeated period headers, metadata header", "src/chunking/hierarchical.py"),
        ("Dense: bge-small-en-v1.5 (fp16 GPU)   |   Sparse: BM25 (fastembed)", "src/embeddings/, src/retrieval/bm25.py"),
        (f"Qdrant: {c['children']:,} children (dense + sparse + payload)\nSQLite parent store: {c['parents_text'] + c['parents_table']:,} parents", "src/retrieval/qdrant_index.py"),
    ]
    y = 11.3
    for i, (t, src) in enumerate(steps):
        box(ax, 0.3, y, 6.4, 1.05, t, fs=8)
        ax.text(6.9, y + 0.52, src, fontsize=7, va="center", color="#555", family="monospace")
        if i < len(steps) - 1:
            arrow(ax, 3.5, y, 3.5, y - 0.35)
        y -= 1.4
    ax.set_title("Offline ingestion and indexing pipeline", fontsize=11, color=ACCENT, loc="left")
    fig.tight_layout(); p = FIG / "fig1_offline_pipeline.png"; fig.savefig(p, dpi=200); plt.close(fig); return p


def fig_online():
    fig, ax = plt.subplots(figsize=(7.2, 9.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13.2); ax.axis("off")
    steps = [
        ("User question", "#FFF2CC"),
        ("Router: comparison / multi-period cue?  no -> original question", "#DCE6F2"),
        ("Decomposition (Qwen3-4B, JSON mode): one lookup per company/metric/period", "#DCE6F2"),
        ("Per sub-question: BM25 top-60 (Qdrant sparse)", "#DCE6F2"),
        ("BGE cross-encoder rerank (fp16, raw logits, tables linearized)", "#DCE6F2"),
        ("Round-robin interleave of sub-question rankings (top 8)", "#DCE6F2"),
        ("Parent expansion within a 5k-token budget (SQLite parent store)", "#DCE6F2"),
        ("CitationQueryEngine: numbered sources + metadata header -> Qwen3-4B (Ollama)", "#DCE6F2"),
        ("<calc> expressions evaluated in Python; every number verified against sources", "#E2EFDA"),
        ("Answer with [n] citations, sources, calculations, unverified numbers", "#FFF2CC"),
    ]
    y = 12.1
    for i, (t, fc) in enumerate(steps):
        box(ax, 0.5, y, 9.0, 0.9, t, fc=fc, fs=8.3)
        if i < len(steps) - 1:
            arrow(ax, 5, y, 5, y - 0.35)
        y -= 1.25
    ax.set_title("Online query pipeline (V5 retrieval + generation)", fontsize=11, color=ACCENT, loc="left")
    fig.tight_layout(); p = FIG / "fig2_online_pipeline.png"; fig.savefig(p, dpi=200); plt.close(fig); return p


def fig_mrr(exp):
    d = exp[exp.dataset == V1].set_index("version")
    order = ["V1", "V1h", "V2", "V3", "V4", "V5", "V5h", "V6"]
    vals = [d.loc[v, "mrr@10"] for v in order]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(order, vals, color=[ACCENT if v in ("V4", "V5") else "#9DB3D6" for v in order])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("MRR@10"); ax.set_ylim(0, 0.85); ax.set_title("Retrieval MRR@10 by version (eval set v1, 236 questions)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = FIG / "fig3_mrr_by_version.png"; fig.savefig(p, dpi=200); plt.close(fig); return p


def fig_allhit(exp):
    d = exp[exp.dataset == V1].set_index("version")
    types = ["numeric", "yoy", "pct_change", "cross_quarter", "cross_company", "narrative"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    w = 0.2
    for i, (v, col) in enumerate(zip(["V1", "V2", "V4", "V5"], ["#C9D6EA", "#9DB3D6", "#5B7DB8", ACCENT])):
        ax.bar([k + (i - 1.5) * w for k in range(len(types))], [d.loc[v, f"{t}_all_hit@10"] for t in types], w, label=v, color=col)
    ax.set_xticks(range(len(types))); ax.set_xticklabels(types, fontsize=8)
    ax.set_ylabel("all-hit@10"); ax.set_ylim(0, 1.08); ax.legend(fontsize=8, ncol=4, frameon=False)
    ax.set_title("Every required fact retrieved in the top 10, by question type (eval v1)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = FIG / "fig4_allhit_by_type.png"; fig.savefig(p, dpi=200); plt.close(fig); return p


def fig_generation(g3, g4):
    types = ["numeric", "yoy", "pct_change", "cross_quarter", "cross_company"]
    fig, ax = plt.subplots(figsize=(7, 3.1))
    w = 0.38
    ax.bar([k - w / 2 for k in range(len(types))], [g3["by_type"].get(t, 0) for t in types], w, label="G3 (no decomposition)", color="#9DB3D6")
    ax.bar([k + w / 2 for k in range(len(types))], [g4["by_type"].get(t, 0) for t in types], w, label="G4 (decomposition)", color=ACCENT)
    ax.set_xticks(range(len(types))); ax.set_xticklabels(types, fontsize=8); ax.set_ylim(0, 1.1)
    ax.set_ylabel("answer correct"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("End-to-end answer accuracy by question type (same 40 questions)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = FIG / "fig5_generation_by_type.png"; fig.savefig(p, dpi=200); plt.close(fig); return p


def fig_scaling(sc):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for v, col in [("V2", "#9DB3D6"), ("V4", "#5B7DB8"), ("V5", ACCENT)]:
        ax.plot(sc.filings, sc[f"{v}_mrr@10"], marker="o", color=col, label=v)
    ax.set_xscale("log"); ax.set_xticks(sc.filings); ax.set_xticklabels([f"{s}\n({n})" for s, n in zip(sc.stage, sc.filings)], fontsize=8)
    ax.set_ylabel("MRR@10"); ax.set_ylim(0.3, 0.8); ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.set_title("Retrieval quality as the corpus grows (stage and filings, log scale)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = FIG / "fig6_scaling.png"; fig.savefig(p, dpi=200); plt.close(fig); return p


# ====================================================================== docx helpers
class Doc:
    def __init__(self):
        self.d = Document()
        sec = self.d.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.2))
        sec.top_margin = sec.bottom_margin = Cm(2.0)
        st = self.d.styles["Normal"]
        st.font.name = "Calibri"; st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.paragraph_format.space_after = Pt(6)
        for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
            h = self.d.styles[f"Heading {lvl}"]
            h.font.name = "Calibri"; h.font.size = Pt(size); h.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)
        self.fig_no = 0
        self.tab_no = 0

    # --- text
    def h(self, text, level=1):
        return self.d.add_heading(arrows(text), level=level)

    def p(self, text="", bold_lead=None, italic=False, size=None, align=None):
        para = self.d.add_paragraph()
        if bold_lead:
            r = para.add_run(arrows(bold_lead)); r.bold = True
        r = para.add_run(arrows(text)); r.italic = italic
        if size:
            r.font.size = Pt(size)
        if align:
            para.alignment = align
        return para

    def bullets(self, items, style="List Bullet"):
        for it in items:
            para = self.d.add_paragraph(style=style)
            if isinstance(it, tuple):
                r = para.add_run(arrows(it[0])); r.bold = True
                para.add_run(arrows(it[1]))
            else:
                para.add_run(arrows(it))

    def source(self, text):
        para = self.d.add_paragraph()
        r = para.add_run(f"Source: {text}"); r.italic = True; r.font.size = Pt(8); r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        para.paragraph_format.space_after = Pt(10)

    def code(self, text):
        for line in text.strip("\n").split("\n"):
            para = self.d.add_paragraph()
            para.paragraph_format.space_after = Pt(0); para.paragraph_format.left_indent = Cm(0.4)
            r = para.add_run(line); r.font.name = "Consolas"; r.font.size = Pt(8.5)
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def callout(self, title, text):
        t = self.d.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = t.rows[0].cells[0]; self._shade(cell, "EAF1FB")
        cell.paragraphs[0].add_run(title).bold = True
        for line in text if isinstance(text, list) else [text]:
            cell.add_paragraph(line).runs[0].font.size = Pt(9.5)
        self.d.add_paragraph()

    # --- tables / figures
    @staticmethod
    def _shade(cell, fill):
        tcPr = cell._element.get_or_add_tcPr()
        shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
        tcPr.append(shd)

    def table(self, header, rows, caption=None, widths=None, font=8.5, numeric_right=True):
        self.tab_no += 1
        if caption:
            cp = self.d.add_paragraph(); r = cp.add_run(f"Table {self.tab_no}. {caption}"); r.bold = True; r.font.size = Pt(9)
            cp.paragraph_format.space_after = Pt(2); cp.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htext in enumerate(header):
            c = t.rows[0].cells[i]; c.text = ""; run = c.paragraphs[0].add_run(str(htext)); run.bold = True; run.font.size = Pt(font)
            self._shade(c, "D9E2F3")
        for row in rows:
            cells = t.add_row().cells
            for i, val in enumerate(row):
                s = arrows(str(val)); cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(s); run.font.size = Pt(font)
                if numeric_right and i > 0 and _is_num(s):
                    cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        return t

    def figure(self, path, caption, width_cm=16):
        self.fig_no += 1
        self.d.add_picture(str(path), width=Cm(width_cm))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp = self.d.add_paragraph(); r = cp.add_run(f"Figure {self.fig_no}. {arrows(caption)}"); r.italic = True; r.font.size = Pt(9)
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def meta(self, goal, built, verdict):
        for lead, text in (("Goal: ", goal), ("Built: ", built), ("Checkpoint verdict: ", verdict)):
            para = self.d.add_paragraph()
            para.paragraph_format.left_indent = Cm(0.4); para.paragraph_format.space_after = Pt(2)
            r = para.add_run(lead); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)
            r2 = para.add_run(arrows(text)); r2.font.size = Pt(9.5)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def page_break(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # --- fields
    @staticmethod
    def _field(paragraph, instr):
        r = paragraph.add_run()
        b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin")
        i = OxmlElement("w:instrText"); i.set(qn("xml:space"), "preserve"); i.text = instr
        s = OxmlElement("w:fldChar"); s.set(qn("w:fldCharType"), "separate")
        t = OxmlElement("w:t"); t.text = "Right-click and choose Update Field (or press F9) to build the table of contents." if "TOC" in instr else "1"
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end")
        for el in (b, i, s, t, e):
            r._r.append(el)

    def toc(self):
        self._field(self.d.add_paragraph(), 'TOC \\o "1-3" \\h \\z \\u')
        # ask Word to refresh fields (TOC) when the document is opened
        settings = self.d.settings.element
        upd = OxmlElement("w:updateFields"); upd.set(qn("w:val"), "true"); settings.append(upd)

    def header_footer(self, title):
        sec = self.d.sections[0]
        hp = sec.header.paragraphs[0]; hr = hp.add_run(title); hr.font.size = Pt(8); hr.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.add_run("Page ").font.size = Pt(8); self._field(fp, "PAGE")

    def save(self, path):
        self.d.save(str(path))


def arrows(text: str) -> str:
    return text.replace(" -> ", " → ").replace("->", "→")


def _is_num(s):
    s = s.replace(",", "").replace("%", "").replace("+", "").replace("-", "").replace(" ms", "").replace(" s", "").strip()
    try:
        float(s); return True
    except ValueError:
        return False


# ====================================================================== content
def build():
    c = R.corpus(); es = R.eval_sets(); exp = R.experiments(); sw = R.sweeps(); sc = R.scaling()
    feas = R.feasibility(); ja = R.judge_agreement(); log = R.git_log(); head = R.head_commit()
    G = {g: R.gen_summary(g) for g in ("G1", "G2", "G3", "G4", "G5", "G6")}
    mc34 = R.mcnemar("G3", "G4")
    e0 = exp[exp.dataset == V0].set_index("version"); e1 = exp[exp.dataset == V1].set_index("version")
    B = lambda a, b, m, ds, sub=None: R.bootstrap(a, b, m, ds, sub)  # noqa: E731
    sig = {
        "v1_v2_v0": B("V1_dense", "V2_bm25", "mrr@10", V0), "v2_v3_v0": B("V2_bm25", "V3_hybrid", "mrr@10", V0),
        "v2_v3_num_v0": B("V2_bm25", "V3_hybrid", "mrr@10", V0, "numeric"),
        "v2_v4_v0": B("V2_bm25", "V4_rerank", "mrr@10", V0), "md_v4_v0": B("V4-md_rerank", "V4_rerank", "mrr@10", V0),
        "md_v4_num_v0": B("V4-md_rerank", "V4_rerank", "mrr@10", V0, "numeric"),
        "v4_v4h_v0": B("V4_rerank", "V4h_rerank", "mrr@10", V0),
        "v4_v5_allhit": B("V4_rerank", "V5_decomposed", "all_hit@10", V1), "v4_v5_mrr": B("V4_rerank", "V5_decomposed", "mrr@10", V1),
        "v1_v1h": B("V1_dense", "V1h_hyde_dense", "mrr@10", V1), "v5h_v6": B("V5h_decomposed", "V6_decomposed", "mrr@10", V1),
        "v5_v6": B("V5_decomposed", "V6_decomposed", "mrr@10", V1), "v5_v6_num": B("V5_decomposed", "V6_decomposed", "mrr@10", V1, "numeric"),
        "s_v5_mrr": B("V5@S24_decomposed", "V5@Sall_decomposed", "mrr@10", V1),
        "s_v5_allhit": B("V5@S24_decomposed", "V5@Sall_decomposed", "all_hit@10", V1),
        "s_v4_mrr": B("V4@S24_rerank", "V4@Sall_rerank", "mrr@10", V1),
        "s_v4_allhit": B("V4@S24_rerank", "V4@Sall_rerank", "all_hit@10", V1),
    }

    def ci(k):
        s = sig[k]
        return f"{s['diff']:+.3f} (95% CI [{s['lo']:+.3f}, {s['hi']:+.3f}], {'significant' if s['sig'] else 'not significant'})"

    figs = {"off": fig_offline(c), "on": fig_online(), "mrr": fig_mrr(exp), "allhit": fig_allhit(exp),
            "gen": fig_generation(G["G3"], G["G4"]), "scale": fig_scaling(sc)}
    D = Doc()
    title = "SEC-EDGAR Financial RAG & Evaluation Engine"
    D.header_footer(f"{title} - Architecture & Engineering Report")

    # ------------------------------------------------------------ title page
    for _ in range(5):
        D.p("")
    D.p(title, size=24, align=WD_ALIGN_PARAGRAPH.CENTER).runs[-1].bold = True
    D.p("Architecture & Engineering Report", size=16, align=WD_ALIGN_PARAGRAPH.CENTER)
    D.p("Retrieval-augmented question answering over Nasdaq-100 SEC 10-K / 10-Q filings, built and measured "
        "phase by phase on a 4 GB-VRAM laptop with free models only.", italic=True, size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(3):
        D.p("")
    D.p(f"Generated {date.today().isoformat()} from the repository state at commit {head}", size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    D.p("Repository: D:\\SEC EDGAR RAG", size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    D.p("Every number in this report is computed from files in the repository by docs/tools/report_data.py; "
        "each table names its source file.", size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    D.page_break()
    D.p("Contents", size=16).runs[-1].bold = True; D.toc(); D.page_break()

    # ------------------------------------------------------------ 1 executive summary
    D.h("1. Executive summary", 1)
    D.p("This project builds a question-answering system over U.S. SEC filings: it retrieves the exact table rows and "
        "passages that answer a financial question and generates a cited answer whose numbers are verified against those "
        "sources. It was built in explicit phases, and every component was kept only if it improved a measured baseline.")
    D.p(f"The corpus is {c['n_filings']} filings ({c['forms'].get('10-K', 0)} annual 10-Ks and {c['forms'].get('10-Q', 0)} "
        f"quarterly 10-Qs, filed {c['filing_dates'][0]} to {c['filing_dates'][1]}) from {c['n_filers_with_filings']} "
        f"Nasdaq-100 companies, parsed into {c['children']:,} retrieval chunks and {c['parents_text'] + c['parents_table']:,} "
        "context sections. The final pipeline splits multi-part questions into single lookups, retrieves each with BM25, "
        "re-scores candidates with a BGE cross-encoder that reads tables as sentences, expands to the parent section, and "
        "answers with a local Qwen3-4B model that delegates all arithmetic to Python.")
    D.p("Headline results (eval set v1: 236 questions with deterministic answer keys, 3 companies; details in Section 6):", bold_lead="")
    D.bullets([
        ("Retrieval MRR@10: ", f"dense {f3(e1.loc['V1', 'mrr@10'])} -> BM25 {f3(e1.loc['V2', 'mrr@10'])} -> + reranker "
                               f"{f3(e1.loc['V4', 'mrr@10'])} -> + decomposition {f3(e1.loc['V5', 'mrr@10'])}."),
        ("Multi-fact questions: ", f"every required fact in the top 10 (all-hit@10) went from {f3(e1.loc['V4', 'cross_quarter_all_hit@10'])} "
                                   f"to {f3(e1.loc['V5', 'cross_quarter_all_hit@10'])} for cross-quarter and "
                                   f"{f3(e1.loc['V4', 'cross_company_all_hit@10'])} to {f3(e1.loc['V5', 'cross_company_all_hit@10'])} for "
                                   "cross-company comparisons once questions were decomposed."),
        ("Table linearization: ", f"showing the reranker table rows as sentences lifted MRR@10 by {ci('md_v4_v0')} on eval v0."),
        ("Answer accuracy: ", f"{pct(G['G3']['correct'])} -> {pct(G['G4']['correct'])} correct on the same 40 questions with "
                              f"decomposition ({mc34['fixed']} fixed, {mc34['broke']} broken, McNemar p = {mc34['p']:.4f})."),
        ("Numerical safety: ", f"{pct(G['G4']['unverified'], 0)} of answers contained a number that no source supports "
                               "(all generation runs G1-G6)."),
        ("Scale: ", f"at {int(sc.iloc[-1].filings)} filings ({int(sc.iloc[-1].points):,} chunks) V5 MRR@10 changed by "
                    f"{ci('s_v5_mrr')} relative to 24 filings; retrieval p50 {sc.iloc[-1]['V5_latency_p50_ms']:.0f} ms."),
        ("Negative results, kept on record: ", "dense retrieval and hybrid fusion did not help on this workload, HyDE gave no "
                                               "measurable gain, and a local 8B model was not a reliable faithfulness judge."),
    ])
    D.p("Constraints that shaped everything: an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM, 16 GB RAM, Windows 11, all data on "
        "the D: drive, and free models only (local open-weight models; hosted judges only on free tiers). Open items: Ragas "
        "scoring is paused at 32 answers because of free-tier limits; numerical-hallucination evaluation, vector quantization, "
        "Phoenix tracing, FastAPI and Docker are the remaining phases.")

    # ------------------------------------------------------------ 1a naming key
    D.h("1a. Naming key: short forms used throughout", 2)
    D.p("Experiments are labelled with short IDs. Each is defined below from the notes and options recorded in "
        "data/eval/results/experiments.csv, the result file names, the generation evaluation code and the commit messages. "
        "Every later chapter spells an ID out on first use.")
    key = [
        ("V1", "Dense retrieval", "bge-small-en-v1.5 embedding search over child chunks, no filters", "v0, v1", "7", "experiments.csv; */V1_dense_*"),
        ("V1h", "Dense retrieval with HyDE", "V1 but the query vector = mean(hypothetical-passage embedding, question embedding)", "v1", "15", "retrieval_v1/V1h_hyde_dense_*"),
        ("V2", "BM25", "Qdrant sparse vectors (fastembed Qdrant/bm25, IDF modifier, avg_len 113.5)", "v0, v1", "8", "*/V2_bm25_*"),
        ("V3", "Hybrid", "dense top-30 + BM25 top-30, relative-score fusion, alpha 0.3 (chosen on dev split)", "v0, v1", "9", "*/V3_hybrid_*"),
        ("V4", "Reranked BM25", "BM25 top-60 -> bge-reranker-base (fp16, raw logits) reading linearized tables", "v0, v1", "10", "*/V4_rerank_*"),
        ("V4-md", "V4 ablation", "same as V4 but the reranker reads markdown tables", "v0", "10", "V4-md_rerank_*"),
        ("V4h", "V4, hybrid pool", "pool = BM25-45 + dense-15, then the V4 reranker", "v0", "10", "V4h_rerank_*"),
        ("V5", "V4 + decomposition", "regex router -> Qwen3-4B JSON sub-questions -> V4 per sub-question -> round-robin interleave", "v1", "14", "retrieval_v1/V5_decomposed_*"),
        ("V5h", "HyDE control", "V5 with pool BM25-45 + plain dense-15", "v1", "15", "retrieval_v1/V5h_decomposed_*"),
        ("V6", "V5 + HyDE", "V5 with pool BM25-45 + HyDE-dense-15", "v1", "15", "retrieval_v1/V6_decomposed_*"),
        ("Vn@Sx", "Scaled run", "retrieval version n measured with the index at scaling stage x (e.g. V5@Sall)", "v1", "16", "retrieval_v1/V*@S*_*"),
        ("G1", "Generation run 1", "30 numeric questions (eval v0), V4 retrieval, prompt v1", "v0", "11", "G1_generation.csv"),
        ("G2", "Generation run 2", "40 questions stratified over 5 answerable types (eval v1), V4, prompt v1", "v1", "12", "G2_generation.csv"),
        ("G3", "Generation run 3", "same 40 questions, prompt v2 (calc for every change, comparisons state the difference, no appended refusal)", "v1", "12", "G3_generation.csv"),
        ("G4", "Generation run 4", "same 40 questions, prompt v2 + decomposition (V5 retrieval)", "v1", "14", "G4_generation.csv"),
        ("G5", "Generation run 5", "G3 setup on 64 questions (40 + 24 narrative); answers and exact contexts saved for Ragas", "v1", "13", "G5_generation.csv, G5_ragas_input.jsonl"),
        ("G6", "Generation run 6", "G4 setup on the same 64 questions; contexts saved for Ragas", "v1", "13", "G6_generation.csv, G6_ragas_input.jsonl"),
        ("S24 ... Sall", "Scaling stages", "index containing 24, 56, 104, 254 and 726 filings (3, 7, 13, 32, 93 companies)", "v1", "16", "scaling_stages.json, scaling.csv"),
        ("v0 / v1", "Evaluation sets", f"v0: {es[V0]['n']} questions (numeric + narrative); v1: {es[V1]['n']} questions (6 types, multi-fact gold groups)", "-", "7, 12", "data/eval/retrieval_v*.jsonl"),
        ("prompt v1 / v2", "Generation prompts", "v2 added rules 4-6: calc for every change incl. rounding, state differences, refusal only if no figure", "-", "11, 12", "src/generation/generator.py (git history)"),
        ("parser v1-v4", "Parser output versions", "v2 section fallbacks; v3 font-size headings + section_mode; v4 header-row detection (year rows, period labels, unit captions)", "-", "3, 10", "src/parsing/parser.py PARSER_VERSION"),
        ("chunker v1-v3", "Chunker output versions", "v2 split oversized footnotes; v3 re-chunk after the header-row fix", "-", "4, 10", "src/chunking/hierarchical.py CHUNKER_VERSION"),
        ("judges", "Ragas judges", "groq-gpt-oss-120b (Groq free plan); local-llama3.1-8b (Ollama, schema-constrained)", "-", "13", "configs/evaluation.yaml"),
    ]
    D.table(["ID", "Stands for", "Exact configuration", "Eval set", "Phase", "Results"], key,
            caption="Run identifiers and version labels", widths=[1.8, 2.6, 6.6, 1.3, 1.1, 3.4], font=7.5, numeric_right=False)
    D.p("Question types (eval v1): numeric (one statement row), yoy (current vs prior-year value in one row), pct_change "
        "(percentage change between two periods), cross_quarter (same metric in two quarterly filings), cross_company "
        "(same concept for two companies), narrative (hand-written, rule-based relevance). Metric short forms are defined "
        "in Section 8 and Appendix C. Result files are named <ID>_<retriever>_per_query.csv and <ID>_<retriever>_traces.jsonl "
        "(for example V4_rerank_per_query.csv, V5@Sall_decomposed_traces.jsonl): eval v0 files sit in data/eval/results/, "
        "eval v1 files in data/eval/results/retrieval_v1/. Ranges such as V1-V3 mean 'V1, V2 and V3'.")
    D.source("data/eval/results/experiments.csv (version, retriever, dataset, notes); src/evaluation/generation.py; git log")

    # ------------------------------------------------------------ 2 constraints
    D.h("2. Constraints and principles", 1)
    D.bullets([
        ("Hardware: ", "RTX 3050 Laptop GPU with 4 GB VRAM, 16 GB RAM, Windows 11. The LLM feasibility benchmark found that "
                       "Windows reserves about 0.8 GB of the GPU and llama.cpp keeps about 1 GB free, so a local LLM gets "
                       "roughly 2.25 GB of VRAM and spills the remaining layers to the CPU (Section 5.6)."),
        ("Storage: ", "all data, models, pip caches and Docker's disk image live on D:; model caches that default to C: "
                      "(Hugging Face, LlamaIndex, fastembed) are redirected in src/config.py."),
        ("Measure every component: ", "each addition is compared against the previous version on a fixed evaluation set with "
                                      "paired significance tests, and settings are chosen on a dev split and reported on a test split."),
        ("Free models only: ", "local open-weight models (bge-small, bge-reranker-base, Qwen3-4B, Llama 3.1 8B) and hosted judges "
                               "only on free plans with call caps and clean stops on quota limits."),
        ("Scale gradually: ", "24 filings end-to-end first, then 56, 104, 254 and 726 filings (Section 5.15)."),
    ])

    # ------------------------------------------------------------ 3 architecture
    D.h("3. System architecture", 1)
    D.h("3.1 Offline ingestion and indexing", 2)
    D.p("The offline pipeline turns public SEC filings into two stores: a Qdrant collection of small, searchable child "
        "chunks (dense vector + BM25 sparse vector + payload), and a SQLite store of the larger parent sections the LLM reads. "
        "Every stage is resumable through JSON checkpoints and idempotent (deterministic IDs; a filing's points are deleted by "
        "accession number before re-upsert).")
    D.figure(figs["off"], "Offline pipeline. Source: src/ingestion, src/parsing, src/chunking, src/embeddings, src/retrieval.", 14.5)
    D.h("3.2 Online query pipeline", 2)
    D.p("At query time a regex router sends only comparison or multi-period questions to the decomposer; each sub-question is "
        "retrieved with BM25 and re-scored by the cross-encoder, and the rankings are interleaved so every fact is represented "
        "near the top. Retrieved children are replaced by their parent sections within a token budget, formatted as numbered "
        "sources and passed with the original question to the local LLM, whose arithmetic and numbers are checked in Python.")
    D.figure(figs["on"], "Online query pipeline. Source: src/query, src/retrieval, src/generation.", 14.5)
    D.h("3.3 Data model", 2)
    D.bullets([
        ("Elements (parser output): ", "heading, paragraph, table (markdown grid + caption + units + footnote links), footnote; "
                                       "each labelled with part, item, section, note and subsection."),
        ("Parents: ", "heading-delimited text groups (<= 1024 tokens) and whole tables with caption, units and footnotes "
                      "(<= 2048 tokens; larger tables split with repeated headers)."),
        ("Children: ", "~256-token sentence-split text, or table row groups that repeat the period header rows."),
        ("Metadata header: ", "company, ticker, form, fiscal period (e.g. 'fiscal 2025 Q3 (quarter ended June 28, 2025)'), heading "
                              "path, table caption and units are prepended to the text the embedder, BM25 and reranker see; IDs and "
                              "URLs are excluded."),
        ("IDs: ", "uuid5(accession/p{i}/c{j}): re-running never duplicates vectors."),
        ("Qdrant collection: ", "named dense vector text-dense (384-d cosine) and sparse vector text-sparse-new (IDF modifier), "
                                "on-disk payload, keyword and datetime payload indexes."),
    ])
    D.h("3.4 Repository map", 2)
    D.table(["Path", "Responsibility"], [
        ("src/ingestion/", "manifest, rate-limited SEC client, resumable downloader, checkpoints, iXBRL verification"),
        ("src/parsing/", "HTML -> elements; sections; table reconstruction; page-artifact removal"),
        ("src/chunking/hierarchical.py", "parent/child TextNodes, metadata header, deterministic IDs"),
        ("src/embeddings/embedder.py", "bge-small embeddings (fp16), per-filing vector cache, token checks"),
        ("src/retrieval/", "Qdrant index, BM25, hybrid fusion, dense/HyDE, linearization, reranker, decomposed retriever, parent store"),
        ("src/query/", "decomposition + router, HyDE"),
        ("src/generation/", "Ollama server control, parent expansion, CitationQueryEngine generator, <calc> + number verification, LLM benchmark"),
        ("src/evaluation/", "eval-set builder, retrieval metrics, sweeps, bootstrap comparison, generation accuracy, Ragas"),
        ("src/pipeline/scale.py", "staged scaling with footprint capture and re-evaluation"),
        ("configs/*.yaml", "every tunable with its rationale"),
        ("tests/", "42 unit tests pinning parser, chunker, retrieval metrics, number safeguards, judge guards"),
    ], caption="Repository layout", widths=[5, 12], numeric_right=False)

    # ------------------------------------------------------------ 4 models
    D.h("4. Models and components: what, why and how they work", 1)
    cfg = R.configs()
    comp = [
        ("4.1 bge-small-en-v1.5 (dense embeddings)",
         [f"Role: bi-encoder for dense retrieval; {cfg['embedding']['dim']}-d vectors, fp16 on GPU, batch {cfg['embedding']['batch_size']}.",
          "Mechanism: query and passage are encoded independently; the CLS representation is L2-normalised so cosine similarity "
          "is a dot product. BGE v1.5 is asymmetric: queries get the instruction prefix 'Represent this sentence for searching "
          "relevant passages:' and passages do not. That asymmetry mattered for HyDE, where the hypothetical passage is embedded "
          "as a passage.",
          "Why: small enough to co-reside with the reranker and the LLM on 4 GB (298 MiB peak during embedding; commit 7fbd8b0), 512-token "
          "limit comfortably above the 490-token largest child. Weakness measured here: a single vector for a 15-row table dilutes "
          "the one row a numeric question asks about (V1 numeric P@3 was 0.024 on eval v0)."]),
        ("4.2 BM25 as Qdrant sparse vectors",
         [f"Role: lexical retrieval; fastembed Qdrant/bm25 with k = {cfg['retrieval']['bm25']['k']}, b = {cfg['retrieval']['bm25']['b']}, "
          f"avg_len = {cfg['retrieval']['bm25']['avg_len']} (measured mean of stemmed, stopword-free chunk length; fastembed's default is 256).",
          "Mechanism: each document stores term-frequency weights with saturation (k) and length normalisation (b); the query sends "
          "each unique stemmed token with weight 1; Qdrant's IDF modifier multiplies by collection-wide IDF at query time, so IDF "
          "stays correct as filings are added.",
          "Why: financial questions name exact line items ('operating income', 'accounts payable'); BM25 lifted numeric P@3 on "
          "eval v0 from 0.024 to 0.264."]),
        ("4.3 bge-reranker-base (cross-encoder)",
         [f"Role: re-scores the ~60 candidates; fp16, batch {cfg['retrieval']['rerank']['batch_size']}.",
          "Mechanism: query and candidate are concatenated and read jointly with full attention (XLM-RoBERTa, 12 layers), so the "
          "model can check that this row is the asked line item for the asked period, which a bi-encoder cannot.",
          "Engineering: ranks on raw logits (the default sigmoid saturates near 0.999 and turns the top of the list into ties); tables "
          "are linearized ('Operating income: Three Months Ended June 28, 2025 = 28,202; ...') because a prose-trained cross-encoder "
          "preferred MD&A text about a metric over the markdown row containing it."]),
        ("4.4 Qdrant and the parent store",
         ["Qdrant v1.19.1 in Docker with a named volume (Windows bind mounts are discouraged by Qdrant); dense vectors use HNSW, "
          "sparse vectors an inverted index; payload indexes on ticker, form, fiscal year/period, content type, item, filing and dates. "
          "Accessed at 127.0.0.1: on Windows 'localhost' tried IPv6 first and stalled every request by about 2 s.",
          "Parents (only fetched by id, never searched) live in SQLite (WAL), which keeps Qdrant's memory for searchable children."]),
        ("4.5 LlamaIndex components",
         ["Used: TextNode relationships (PARENT/CHILD/PREVIOUS/NEXT/SOURCE), SentenceSplitter, HuggingFaceEmbedding, QdrantVectorStore "
          "(hybrid mode with custom sparse functions and fusion), SentenceTransformerRerank (subclassed), CitationQueryEngine, "
          "HyDEQueryTransform and the Ollama LLM.",
          "Deliberately bypassed: default HTML readers (tables would lose structure), HierarchicalNodeParser (not structure-aware), "
          "the reranker's sigmoid output and markdown input, the embedding of HyDE passages with the query instruction, and "
          "SubQuestionQueryEngine (4-5 LLM calls per question instead of 2)."]),
        ("4.6 Qwen3-4B-Instruct-2507 via Ollama (generator)",
         ["Q4_K_M quantization, q8_0 KV cache, flash attention, 8k context, temperature 0, one parallel slot. Only ~2.25 GB of the "
          "model fits on the GPU; the rest runs on the CPU.",
          "Why: in the feasibility benchmark it was the only candidate that quoted both correct values in all six configurations "
          "with no unsupported numbers (Section 5.6). Arithmetic is delegated to Python through <calc> tags because every observed "
          "error in the benchmark was a subtraction or a misread value."]),
        ("4.7 Judges for Ragas",
         ["groq-gpt-oss-120b: open-weight 120B model on Groq's free plan (no billing), reasoning effort low; not the Qwen family, so "
          "the generator does not grade itself. Limits observed on this plan: 8,000 tokens/minute (logs/ragas_full.log) and "
          "~200K tokens/day (README).",
          "local-llama3.1-8b: Ollama, schema-constrained decoding (free JSON mode made it echo the schema). Validated against the "
          "120B judge before use (Section 5.13)."]),
        ("4.8 Ragas 0.4 metrics",
         ["Faithfulness: the judge extracts the answer's claims and checks each against the retrieved contexts; score = supported / total. "
          "A refusal has no claims and would score 0, so refusals are excluded from judging.",
          "Answer relevancy: the judge generates a question from the answer and its embedding similarity to the real question is the "
          "score (local bge-small). Terse answers such as '$12,016 million [1].' score low even when correct, so the metric is used to "
          "compare setups, not as an absolute quality score."]),
    ]
    for title_, paras in comp:
        D.h(title_, 2)
        for t in paras:
            D.p(t)
    D.source("configs/*.yaml; src/retrieval/reranker.py; src/generation/generator.py; README.md")

    # ------------------------------------------------------------ 5 chronology
    D.h("5. Chronological account of the phases", 1)
    D.p("The project followed a frozen plan with checkpoints (data, retrieval, reranking, generation, production). Two "
        "deviations are recorded: a local-LLM feasibility benchmark was inserted before generation, and sub-question "
        "decomposition (Phase 14) was done before Ragas (Phase 13) because multi-fact questions were the largest measured gap.")
    D.table(["Commit", "Date", "Subject"], log.values.tolist(), caption="Commit history (chronology)", widths=[1.6, 2.8, 12.6],
            font=7.5, numeric_right=False)
    D.source("git log --reverse")

    D.h("5.1 Phases 0-2: setup and corpus acquisition (86d486a)", 2)
    D.meta('a reproducible, verified corpus of 10-K/10-Q primary documents on D:.', 'src/ingestion/{metadata,sec_client,downloader,checkpoint,verify}.py; data/metadata/*.csv', 'passed - every downloaded file matches its manifest entry on form type, period end and CIK.')
    D.p(f"Goal: a reliable, verifiable corpus. The manifest is built from the Wikipedia Nasdaq-100 list mapped to SEC CIKs "
        f"({c['n_companies_listed']} filers after merging share classes such as GOOG/GOOGL) and the SEC submissions API, keeping the "
        f"latest 2 10-Ks and 6 10-Qs per filer: {c['n_filings']} filings. {len(c['foreign'])} foreign private issuers "
        f"({', '.join(c['foreign'])}) file 20-F/40-F and are excluded by design. Downloads are rate-limited (5 req/s, under the "
        f"SEC's 10), validated, written atomically with metadata sidecars and checkpointed: {c['raw_gb']:.2f} GB of primary documents.")
    D.p(f"Verification reads each document's own iXBRL cover tags (dei:DocumentType, DocumentPeriodEndDate, EntityCentralIndexKey) "
        f"and compares them with the manifest: {c['verified_ok']}/{c['verified_total']} match. It also records the fiscal year and "
        "period, which later drive period labels (NVIDIA's quarter ending July 2026 is fiscal 2027 Q2).")
    D.p("Problems fixed: nested ix:nonNumeric tags (a DocumentPeriodEndDate wrapping a CurrentFiscalYearEndDate) produced false "
        "mismatches until extraction became nesting-aware; the checkpoint's atomic rename was refused by transient Windows file "
        "locks, fixed with short retries.")
    D.source("data/metadata/*.csv; src/ingestion/verify.py; src/ingestion/checkpoint.py")

    D.h("5.2 Phase 3: structure-preserving parser (2cd034a)", 2)
    D.meta('turn typeset HTML into ordered, labelled elements without breaking tables.', 'src/parsing/{parser,structure,tables,cleaners}.py; tests/test_parsing.py', 'passed with a documented limitation - standard Item structure for the large majority of filings, title fallback flagged for the rest.')
    D.p(f"SEC HTML is typeset, not structured. The parser walks the DOM into text blocks, tables and page breaks; removes running "
        f"headers/footers and page numbers near page breaks; tracks Part/Item/Note headings with ordering guards; and rebuilds tables "
        f"(colspan expansion, '$' and ')' cells merged into values, compatible-column merging, header detection, continuation tables "
        f"inheriting headers, captions, units and footnotes). Across the corpus: {c['n_tables']:,} tables and {c['n_footnotes']:,} "
        f"linked footnotes; {c['section_mode'].get('items', 0)} filings use standard Item headings and {c['section_mode'].get('titles', 0)} "
        "(Intel, Honeywell 10-Ks) needed a title-based fallback, flagged as lower confidence.")
    D.p("Problems fixed: <br> and block boundaries concatenated words ('June 28,2025'); spanned period headers were merged into the label "
        "column (Microsoft balance sheet); split tables lost their period headers; 10-Q Item 1 headings were often missing; Intel "
        "marks headings with font size rather than bold. Later (Phase 10) the header detector was extended to year rows, period labels "
        "in the label column and unit captions (parser v4).")
    D.source("data/checkpoints/parse_state.json; src/parsing/*.py; tests/test_parsing.py")

    D.h("5.3 Phase 4: parent-child chunking (839bb3c)", 2)
    D.meta('small precise retrieval units that still carry their table headers, period and section.', 'src/chunking/hierarchical.py; configs/chunking.yaml; tests/test_chunking.py', "passed - every child within the embedder's 512-token limit; every table child repeats its period header.")
    D.p(f"LlamaIndex TextNodes with parent/child relationships: {c['children']:,} children (median embedding text "
        f"{c['child_tok_p50']:.0f} tokens, largest {c['child_tok_max']}) and {c['parents_text']:,} text + {c['parents_table']:,} table "
        "parents. Table children are row groups that repeat the period header, so '| Operating income | 28,202 | 25,352 |' always "
        "arrives with 'Three Months Ended June 28, 2025 / June 29, 2024'.")
    D.p("Problems fixed: children inside merged parents carried the first subheading instead of their own (fixed by labelling each child "
        "at its midpoint); some footnote blocks exceeded the 512-token embedder limit (split at sentence boundaries).")
    D.source("data/checkpoints/chunk_state.json; src/chunking/hierarchical.py; tests/test_chunking.py")

    D.h("5.4 Phase 5: dense embeddings (7fbd8b0)", 2)
    D.meta('dense vectors on a 4 GB GPU, cached for reuse.', 'src/embeddings/embedder.py; configs/embedding.yaml', 'passed as infrastructure; dense retrieval itself became the weak baseline (V1).')
    D.p("bge-small-en-v1.5 in fp16 on the GPU with per-filing vector caching, so indexing and model experiments never recompute. "
        "The dense baseline exposed the central weakness: for numeric questions the top-3 results were always the right company, "
        "but only 6% of those chunks contained the asked line item, because one vector per table chunk dilutes a single row (README, V1 finding).")
    D.p("Problem fixed: LlamaIndex's HuggingFaceEmbedding ignored HF_HOME and cached the model on C:; LLAMA_INDEX_CACHE_DIR now points to D:.")

    D.h("5.5 Phase 6: Qdrant index (ef8ebde)", 2)
    D.meta('a persistent vector store ready for hybrid search, plus a parent store.', 'docker-compose.yml; src/retrieval/{qdrant_index,parent_store}.py; configs/qdrant.yaml', 'passed - Qdrant results match brute force; re-indexing is idempotent.')
    D.p("A single collection with named dense and (reserved) sparse vectors in LlamaIndex's naming, so hybrid search later needed no "
        "migration; payload indexes for filters; SQLite parent store. Qdrant's top-10 matched brute-force search (same sets; only "
        "score ties swapped).")
    D.p("Bottleneck: requests to 'localhost' took about 2 s each on Windows (IPv6 tried first, port bound to IPv4); switching to "
        "127.0.0.1 cut per-request latency to about 8 ms and indexing the 24 test filings from 3 min 46 s to about 21 s (README).")

    D.h("5.6 Local-LLM feasibility on 4 GB VRAM (903d9f9)", 2)
    D.meta('choose a generator that fits 4 GB VRAM next to the embedder and reranker.', 'src/generation/{ollama_server,llm_feasibility}.py; configs/llm.yaml', 'Qwen3-4B selected on correctness; speed and context length are the binding constraints.')
    fsum = (feas.assign(both=lambda d: d.has_2025_value & d.has_2024_value,
                        unsupported=lambda d: d.unsupported_numbers.fillna("").astype(str).str.len() > 0)
            .groupby("model").agg(runs=("model", "size"), correct=("both", "sum"), wrong=("unsupported", "sum")))
    speed = feas[feas.kv_cache == "q8_0"].pivot(index="model", columns="context", values="gen_tok_s")
    rows = [(m, f"{speed.loc[m, 4096]:.1f}", f"{speed.loc[m, 8192]:.1f}", f"{speed.loc[m, 16384]:.1f}",
             f"{int(fsum.loc[m, 'correct'])}/{int(fsum.loc[m, 'runs'])}", f"{int(fsum.loc[m, 'wrong'])}/{int(fsum.loc[m, 'runs'])}")
            for m in fsum.index]
    D.p("Three 4-bit (Q4_K_M) candidates were run on a real RAG prompt (Apple FY2025 Q3 parents) at 4k/8k/16k context with f16 and q8_0 "
        "KV caches, recording VRAM, CPU spill, speed and whether the answer quoted the correct figures.")
    D.table(["Model", "gen tok/s 4k", "8k", "16k", "both values correct", "runs with unsupported numbers"], rows,
            caption="LLM feasibility (q8_0 KV cache for speed; correctness over all 6 runs per model)", widths=[3.2, 2.3, 1.6, 1.6, 3.2, 4.4])
    D.source("data/benchmarks/llm_feasibility.csv")
    D.p("Qwen3-4B was chosen: slower than Llama-3.2-3B but the only model with every run correct and no unsupported numbers. With the "
        "embedder and reranker resident (927 MiB), all three still fit (3.3 of 4.0 GB used; README co-residency table).")

    D.h("5.7 Phase 7: evaluation set v0 and the dense baseline (c918f46)", 2)
    D.meta('a deterministic, auditable evaluation set and a measured baseline.', 'src/evaluation/{dataset,retrieval}.py; configs/eval_narrative.yaml; src/retrieval/dense.py', 'baseline established (checkpoint 1) - and shown to be weak on table rows.')
    D.p(f"Eval set v0: {es[V0]['n']} questions ({es[V0]['types'].get('numeric', 0)} numeric generated from primary-statement rows with "
        f"the period text validated against the filing's period end; {es[V0]['types'].get('narrative', 0)} hand-written narrative "
        "questions with auditable relevance rules). Dense retrieval (V1) found the right company but rarely the asked row.")

    D.h("5.8 Phases 8-9: BM25 and hybrid retrieval (1124f0b)", 2)
    D.meta('add lexical matching and test whether fusing it with dense search helps.', 'src/retrieval/{bm25,hybrid}.py; src/evaluation/fusion_sweep.py; configs/retrieval.yaml', 'BM25 adopted; hybrid fusion not adopted for ranking (no gain), kept only as a candidate-pool option.')
    fs = sw["fusion_sweep"]
    D.p(f"BM25 (V2) moved MRR@10 on v0 by {ci('v1_v2_v0')} over dense. Hybrid fusion was tuned on the dev split: relative-score fusion with "
        f"alpha 0.3 (dev MRR {fs[(fs.fusion == 'relative') & (fs.alpha.astype(str) == '0.3') & (fs.split == 'dev')]['mrr@10'].iloc[0]:.3f}) "
        f"beat alpha 0.5, 0.7 and RRF (dev MRR {fs[(fs.fusion == 'rrf') & (fs.split == 'dev')]['mrr@10'].iloc[0]:.3f}). Even so, hybrid V3 "
        f"did not beat BM25: V2 -> V3 MRR {ci('v2_v3_v0')}, numeric {ci('v2_v3_num_v0')}.")
    cp = sw["candidate_pools"].set_index("pool")
    D.p(f"Its value was as a candidate pool: BM25 top-60 contained a relevant chunk for {pct(cp.loc['bm25 top-60', 'hit'])} of "
        f"questions and dense top-60 for {pct(cp.loc['dense top-60', 'hit'])}; only the union reached "
        f"{pct(cp.loc['union dense30+bm25-30', 'hit_narrative'])} on narrative questions.")
    D.source("data/eval/results/fusion_sweep.csv, candidate_pools.csv (both measured before the parser v4 header fix)")

    D.h("5.9 Phase 10: cross-encoder reranking (8932d01)", 2)
    D.meta('precision on top of a high-recall pool using a cross-encoder.', 'src/retrieval/{reranker,linearize}.py; src/evaluation/compare.py', 'passed (checkpoint 3) - significant gain over BM25 once tables are linearized.')
    rs = sw["rerank_sweep"]
    md_test = rs[(rs.pool == "bm25_60") & (~rs.linearize) & (rs.split == "test")]["mrr@10"].iloc[0]
    lin_test = rs[(rs.pool == "bm25_60") & (rs.linearize) & (rs.split == "test")]["mrr@10"].iloc[0]
    D.p(f"Out of the box the reranker made numeric retrieval worse: it preferred MD&A prose about a metric to the markdown row containing "
        f"it (for 22 of 40 inspected numeric questions the top-1 result was such a text chunk; README). Two fixes: rank "
        f"on raw logits, and linearize tables into sentences. On the test split, BM25-60 + reranker went from MRR {md_test:.3f} (markdown) "
        f"to {lin_test:.3f} (linearized). Final V4 vs V2 on v0: {ci('v2_v4_v0')}; linearization alone (V4-md -> V4): {ci('md_v4_v0')}, "
        f"numeric {ci('md_v4_num_v0')}; adding dense candidates (V4 -> V4h): {ci('v4_v4h_v0')}.")
    D.p("A second finding came from diagnosing the reranker: Microsoft tables stored their year row ('2025 | 2024') and period label "
        "('Year Ended June 30,') in forms the header detector missed, so later row-group chunks did not repeat their periods. The "
        "fix (parser v4, chunker v3) improved BM25 on its own (V2 v0 MRR moved to its current value in experiments.csv).")
    D.source("data/eval/results/rerank_sweep.csv; experiments.csv (v0); per-query files in data/eval/results/")

    D.h("5.10 Phase 11: grounded generation (a5dfc76)", 2)
    D.meta('grounded, cited answers whose numbers can be trusted.', 'src/generation/{generator,context,numbers}.py; src/evaluation/generation.py; configs/generation.yaml', 'passed (checkpoint 4) - no unsupported numbers; remaining errors traced to question generation.')
    g1 = G["G1"]
    D.p(f"CitationQueryEngine over V4 retrieval with parent expansion, the local Qwen3-4B and a grounding prompt. G1 (30 numeric questions): "
        f"{pct(g1['correct'])} correct, gold in context {pct(g1['ctx'])}, {pct(g1['unverified'], 0)} answers with unverified numbers, "
        f"p50 {g1['p50_s']:.1f} s. Most misses traced to badly generated questions (fixed in Phase 12).")
    D.p("Bottlenecks: prompt processing collapsed to about 70 tokens/s because orphaned llama-server.exe runners from earlier runs held "
        "VRAM and pushed the GPU into shared-memory spill; stop() now kills them, the reranker warms up before the LLM loads, and the "
        "reranker batch is 16 (439 ms / 906 MiB for 60 pairs vs 939 ms / 1218 MiB at 32; configs/retrieval.yaml).")

    D.h("5.11 Phase 12: evaluation set v1 (cc315ae)", 2)
    D.meta('an evaluation set that covers every question type in the plan, including multi-fact questions.', 'src/evaluation/dataset.py (v1); data/eval/retrieval_v1.jsonl', 'exposed the largest remaining gap: multi-fact retrieval at 0.000 all-hit@10.')
    t = es[V1]["types"]
    D.p(f"Eval set v1: {es[V1]['n']} questions - {t.get('numeric', 0)} numeric, {t.get('yoy', 0)} yoy, {t.get('pct_change', 0)} pct_change, "
        f"{t.get('cross_quarter', 0)} cross_quarter, {t.get('cross_company', 0)} cross_company, {t.get('narrative', 0)} narrative - with one "
        "gold group per required fact and new metrics all-hit@k and fact_recall@10. Generator fixes: cash-flow working-capital rows are "
        "phrased as 'change in ...', groups end at their total or 'Net cash' row, current balance-sheet items are marked '(current)', % change "
        "only for positive levels.")
    D.p(f"Every retriever scored 0.000 all-hit@10 on cross-quarter and cross-company questions: the compound question pulled MD&A comparison "
        f"text instead of the statement rows. Prompt v2 then fixed generation behaviours (G2 -> G3): correct {pct(G['G2']['correct'])} -> "
        f"{pct(G['G3']['correct'])}, correct given facts in context {pct(G['G2']['correct_given_ctx'])} -> {pct(G['G3']['correct_given_ctx'])}, "
        f"wrong refusals {G['G2']['bad_refusals']} -> {G['G3']['bad_refusals']}.")

    D.h("5.12 Phase 14: sub-question decomposition (fb67546)", 2)
    D.meta('answer comparisons by retrieving each fact separately.', 'src/query/decomposition.py; src/retrieval/decomposed.py', 'adopted - the largest single improvement in the project.')
    D.p(f"One Qwen call in JSON mode splits comparisons into single lookups (few-shot examples use companies not in the eval set); a single "
        f"returned sub-question falls back to the original wording; a regex router agreed with the LLM's own split decisions on all eval "
        f"questions and removed the LLM call from single-fact questions (median retrieval latency 2.3 s -> {e1.loc['V5', 'latency_p50_ms'] / 1000:.1f} s). "
        f"V4 -> V5 all-hit@10: {ci('v4_v5_allhit')}; MRR {ci('v4_v5_mrr')}. Generation G3 -> G4: {pct(G['G3']['correct'])} -> "
        f"{pct(G['G4']['correct'])} ({mc34['fixed']} fixed, {mc34['broke']} broken, McNemar p = {mc34['p']:.4f}).")
    D.figure(figs["gen"], "Answer accuracy by type, G3 vs G4. Source: data/eval/results/G3_generation.csv, G4_generation.csv.", 15)

    D.h("5.13 Phase 13: Ragas with free judges (a9c8ed8, ef28e23) - paused", 2)
    D.meta('LLM-judged faithfulness and relevancy with free judges only.', 'src/evaluation/ragas_eval.py; configs/evaluation.yaml; tests/test_ragas_guard.py', 'paused by decision - infrastructure complete; 32 G5 answers judged; local judge valid for relevancy only.')
    D.p(f"G5 and G6 re-generated the 40 questions plus 24 narrative ones and saved the exact contexts; they reproduce G3 and G4 exactly "
        f"({pct(G['G5']['correct'])} and {pct(G['G6']['correct'])}), confirming determinism. The 120B judge scored {ja['n'] + ja['refused']} "
        f"G5 answers ({ja['refused']} refusals excluded) before Groq's free daily token limit stopped the run cleanly.")
    D.table(["Metric", "120B judge", "8B judge", "mean |diff|", "Pearson", "Spearman", "same side of 0.5"],
            [(m, f3(ja[m]["a"]), f3(ja[m]["b"]), f3(ja[m]["mad"]), f"{ja[m]['pearson']:.2f}", f"{ja[m]['spearman']:.2f}",
              pct(ja[m]["same_side"], 0)) for m in ("faithfulness", "answer_relevancy")],
            caption=f"Judge agreement on the same {ja['n']} G5 answers", widths=[3.2, 2, 2, 2, 1.8, 1.8, 2.6])
    D.source("data/eval/results/G5_ragas_groq-gpt-oss-120b.csv, G5_ragas_local-llama3.1-8b.csv")
    D.p(f"The local 8B judge agrees on relevancy but not faithfulness; in the disagreements ({', '.join(ja['flips'])}) the deterministic "
        "answer key shows the answers were correct, so it is used for relevancy only. Engineering issues solved on the way: ragas "
        "0.4.3 needed langchain-community 0.4.1, instructor >= 1.9 and openai < 3; the async client had to live in a single event loop; "
        "free-tier per-minute limits are waited out while daily limits stop the run; each judge has its own cache.")

    D.h("5.14 Phase 15: HyDE, evaluated with controls (4f27d78)", 2)
    D.meta('test whether HyDE improves retrieval, isolated from the effect of adding dense candidates.', 'src/query/hyde.py; src/retrieval/dense.py (retrieve_hyde); pool union_45_hyde15', 'not adopted - no significant gain, about 3 s extra latency.')
    D.p(f"A <= 80-word filing-style passage is generated by HyDEQueryTransform (local Qwen), embedded as a passage and averaged with the "
        f"query embedding. V1 -> V1h: {ci('v1_v1h')}. V5h -> V6 (same pool, plain vs HyDE dense): {ci('v5h_v6')}. V5 -> V6: {ci('v5_v6')}; "
        f"numeric {ci('v5_v6_num')}. HyDE costs about 3 s per query and is off by default.")

    D.h("5.15 Phase 16: scaling to all filings (aa554e4)", 2)
    D.meta('confirm quality, latency and footprint hold on the full corpus.', 'src/pipeline/scale.py; data/eval/scaling_stages.json', 'passed - V5 quality change not significant at 30x the corpus; latency flat.')
    D.p(f"Stages add companies with the eval companies first and re-run V2, V4 and V5 on the unchanged eval set. V5 from S24 to Sall: MRR "
        f"{ci('s_v5_mrr')}, all-hit@10 {ci('s_v5_allhit')}. V4 (no decomposition): MRR {ci('s_v4_mrr')}, all-hit@10 {ci('s_v4_allhit')}.")
    rows = [(r.stage, int(r.filings), int(r.companies), f"{int(r.points):,}", f3(r["V2_mrr@10"]), f3(r["V4_mrr@10"]), f3(r["V5_mrr@10"]),
             f3(r["V5_all_hit@10"]), f"{r['V5_latency_p50_ms']:.0f}", r.qdrant_mem, r.qdrant_disk, f"{r.parents_db_mb:.0f}")
            for _, r in sc.iterrows()]
    D.table(["Stage", "Filings", "Cos.", "Chunks", "V2 MRR", "V4 MRR", "V5 MRR", "V5 all-hit", "V5 p50 ms", "Qdrant RAM", "Qdrant disk", "Parents MB"],
            rows, caption="Scaling stages", font=7.5, widths=[1.2, 1.2, 1, 1.6, 1.3, 1.3, 1.3, 1.4, 1.4, 1.8, 1.5, 1.4])
    D.source("data/eval/results/scaling.csv; data/eval/scaling_stages.json")
    D.figure(figs["scale"], "MRR@10 across scaling stages. Source: data/eval/results/scaling.csv.", 15)

    # ------------------------------------------------------------ 6 ledger
    D.h("6. Experiment ledger", 1)
    D.h("6.1 Retrieval on eval v0 (164 questions)", 2)
    cols = ["precision@3", "recall@5", "mrr@10", "hit@10", "numeric_precision@3", "narrative_precision@3", "latency_p50_ms"]
    rows = [(v, *[f3(e0.loc[v, k]) if k != "latency_p50_ms" else f"{e0.loc[v, k]:.0f}" for k in cols]) for v in ["V1", "V2", "V3", "V4-md", "V4", "V4h"]]
    D.table(["Run", "P@3", "R@5", "MRR@10", "Hit@10", "numeric P@3", "narrative P@3", "p50 ms"], rows, caption="Retrieval, eval v0")
    D.source("data/eval/results/experiments.csv (dataset = retrieval_v0.jsonl)")
    D.p("The P@3 ceiling on v0 is 0.632 because most questions have only one or two relevant chunks (README); V4 reaches about 70% of it.")
    D.h("6.2 Retrieval on eval v1 (236 questions)", 2)
    cols = ["precision@3", "recall@5", "mrr@10", "hit@10", "all_hit@10", "latency_p50_ms"]
    rows = [(v, *[f3(e1.loc[v, k]) if k != "latency_p50_ms" else f"{e1.loc[v, k]:.0f}" for k in cols]) for v in ["V1", "V1h", "V2", "V3", "V4", "V5", "V5h", "V6"]]
    D.table(["Run", "P@3", "R@5", "MRR@10", "Hit@10", "all-hit@10", "p50 ms"], rows, caption="Retrieval, eval v1")
    D.source("data/eval/results/experiments.csv (dataset = retrieval_v1.jsonl)")
    D.figure(figs["mrr"], "MRR@10 by version on eval v1. Source: data/eval/results/experiments.csv.", 15)
    D.figure(figs["allhit"], "all-hit@10 by question type. Source: data/eval/results/experiments.csv.", 15.5)
    D.h("6.3 Significance of key comparisons (paired bootstrap, 10,000 resamples)", 2)
    sig_rows = [("V1 -> V2 (v0, MRR)", "v1_v2_v0"), ("V2 -> V3 (v0, MRR)", "v2_v3_v0"), ("V2 -> V4 (v0, MRR)", "v2_v4_v0"),
                ("V4-md -> V4 (v0, MRR)", "md_v4_v0"), ("V4 -> V4h (v0, MRR)", "v4_v4h_v0"), ("V4 -> V5 (v1, all-hit@10)", "v4_v5_allhit"),
                ("V4 -> V5 (v1, MRR)", "v4_v5_mrr"), ("V1 -> V1h (v1, MRR)", "v1_v1h"), ("V5h -> V6 (v1, MRR)", "v5h_v6"),
                ("V5 -> V6 (v1, MRR)", "v5_v6"), ("V5@S24 -> V5@Sall (MRR)", "s_v5_mrr"), ("V4@S24 -> V4@Sall (MRR)", "s_v4_mrr")]
    D.table(["Comparison", "A", "B", "diff", "95% CI", "significant"],
            [(n, f3(sig[k]["a"]), f3(sig[k]["b"]), f"{sig[k]['diff']:+.3f}", f"[{sig[k]['lo']:+.3f}, {sig[k]['hi']:+.3f}]",
              "yes" if sig[k]["sig"] else "no") for n, k in sig_rows],
            caption="Paired bootstrap over questions (recomputed from per-query files)", widths=[5.2, 1.6, 1.6, 1.6, 3.8, 2.2])
    D.source("per-query files in data/eval/results/ and data/eval/results/retrieval_v1/; method of src/evaluation/compare.py")
    D.callout("Inconsistencies noted between sources", [
        "README experiment values for V1-V3 were first hand-copied and partly wrong; commit de5be8a regenerated the table from "
        "experiments.csv, which this report uses.",
        "fusion_sweep.csv and candidate_pools.csv were measured before the parser v4 / chunker v3 header fix; experiments.csv v0 rows "
        "(V1-V3 re-run) and rerank_sweep.csv were measured after it, so sweep numbers are not directly comparable with Table 7.",
        "experiments.csv rows from before eval set v1 had no dataset value; they were tagged retrieval_v0.jsonl when v1 was introduced.",
        "data/eval/cache/decompositions.jsonl contains a cached decomposition of the internal string 'warm-up query'; it is never part "
        "of an evaluation.",
    ])
    D.h("6.4 Generation", 2)
    D.table(["Run", "n", "correct", "facts in context", "correct | context", "refused", "wrong refusals", "unverified numbers", "cited", "p50 s"],
            [(g, s["n"], pct(s["correct"]), pct(s["ctx"]), pct(s["correct_given_ctx"]), pct(s["refused"]), s["bad_refusals"],
              pct(s["unverified"], 0), pct(s["cited"]), f"{s['p50_s']:.1f}") for g, s in G.items()],
            caption="Generation runs (answer-keyed questions only)", font=7.5)
    D.source("data/eval/results/G1..G6_generation.csv; McNemar G3 -> G4 computed from G3/G4 files")

    # ------------------------------------------------------------ 7 bottlenecks
    D.h("7. Bottlenecks, bugs and fixes", 1)
    bugs = [
        ("Checkpoint rename refused (Windows file locks)", "download crashed mid-run", "antivirus/editor locks the target briefly", "retry os.replace", "src/ingestion/checkpoint.py"),
        ("Nested iXBRL tags", "false period mismatches in verification", "regex stopped at inner closing tag", "nesting-aware extraction", "src/ingestion/verify.py"),
        ("Words glued ('June 28,2025')", "table inspection", "<br>/block boundaries dropped by text_content", "insert spaces at block tails", "src/parsing/parser.py"),
        ("Period header in label column", "MSFT balance sheet misaligned", "column merge treated spanned headers as compatible", "header rows never merge into label column", "src/parsing/tables.py"),
        ("Split tables lost headers", "Apple % table without periods", "continuation tables typeset separately", "inherit caption/units/header rows", "src/parsing/parser.py"),
        ("Missing / non-standard section headings", "corpus-wide section stats", "10-Q Item 1 omitted; Intel/HON layouts", "Part I default, title fallback, font-size headings", "src/parsing/structure.py"),
        ("Children with wrong subheading", "Gross Margin text labelled otherwise", "merged parents kept first heading", "label child at its midpoint", "src/chunking/hierarchical.py"),
        ("Oversized footnote chunks", "children over the 512-token embedder limit", "all footnotes in one child", "sentence-split footnotes", "src/chunking/hierarchical.py"),
        ("Model caches on C:", "downloads landed in user profile", "HF / LlamaIndex / fastembed defaults", "cache env vars point to D:", "src/config.py"),
        ("localhost ~2 s per request", "indexing 3m46s for 24 filings", "Windows tries IPv6 first", "use 127.0.0.1 (~8 ms)", "configs/qdrant.yaml"),
        ("Reranker ties at 0.999", "saturated scores", "sigmoid on strong matches", "rank on raw logits", "src/retrieval/reranker.py"),
        ("Reranker preferred prose", "correct row rarely first", "prose-trained cross-encoder vs markdown", "linearize tables for the reranker", "src/retrieval/linearize.py"),
        ("Year rows not headers (MSFT)", "row groups without periods", "header detector too strict", "year rows/period labels/unit captions are headers", "src/parsing/tables.py"),
        ("Prompt processing ~70 tok/s", "generation eval far slower than expected", "orphaned llama-server.exe holding VRAM", "kill runners; warm-up order; batch 16", "src/generation/ollama_server.py"),
        ("Compound questions break BM25", "cross-type all-hit@10 = 0.000", "comparison wording matches MD&A text", "sub-question decomposition + router", "src/query/decomposition.py"),
        ("Flawed generated questions", "G1 misses", "cash-flow change rows, group labels", "'change in ...', groups end at totals", "src/evaluation/dataset.py"),
        ("Rewording lost intent", "'What drove X' -> 'What was X'", "LLM rewrote single lookups", "keep original if one sub-question", "src/query/decomposition.py"),
        ("Dependency conflicts", "ragas import errors", "ragas 0.4.3 vs langchain/instructor/openai", "pin langchain-community 0.4.1, instructor, openai<3", "requirements.txt"),
        ("Async connection errors", "judge calls failing", "new event loop per metric", "one event loop per run", "src/evaluation/ragas_eval.py"),
        ("Free-tier limits", "per-minute and daily 429s", "Groq free plan quotas", "wait on per-minute, stop on daily, cache", "src/evaluation/ragas_eval.py"),
        ("Small judge echoed schema", "local 8B output invalid", "free JSON mode on long prompts", "schema-constrained decoding", "configs/evaluation.yaml"),
    ]
    D.table(["Problem", "How it showed up", "Root cause", "Fix", "Where"], bugs, caption="Bottlenecks and fixes (chronological)",
            widths=[3.4, 3.3, 3.6, 3.6, 3.4], font=7.2, numeric_right=False)
    D.source("commit messages; code comments at the listed paths; README.md")

    # ------------------------------------------------------------ 8 methodology
    D.h("8. Evaluation methodology", 1)
    D.bullets([
        ("Deterministic gold: ", "numeric questions come from primary-statement rows; a chunk is relevant if it contains that row with that "
                                 "value (the same figure reappears as the comparative column of the next year's filing) or a text chunk "
                                 "states the value with the line item. Narrative relevance uses explicit rules in configs/eval_narrative.yaml."),
        ("Periods: ", "the period text of the value column is validated against the filing's period end before a question is emitted."),
        ("Metrics: ", "P@3; R@5 capped at min(|gold|, 5); Hit@k; MRR@10; nDCG@10; for multi-fact questions all-hit@k (every fact found) "
                      "and fact_recall@10."),
        ("Selection vs reporting: ", "a deterministic 50/50 dev/test split by question id; fusion and pool settings were chosen on dev."),
        ("Significance: ", "paired bootstrap over questions (10,000 resamples, 95% CI); McNemar for paired correctness."),
        ("Why not an LLM judge for retrieval: ", "deterministic gold is reproducible, free and stricter than a judge."),
        ("Limitations: ", "3 companies, 85% numeric in v0; narrative sets are small (24); relevancy scores penalise terse answers."),
    ])

    # ------------------------------------------------------------ 9 perspective
    D.h("9. Results in perspective", 1)
    D.p("Why dense retrieval and HyDE did not help: questions name exact line items and periods, which BM25 matches directly, while a single "
        "vector per table chunk averages over many rows. Once a cross-encoder reads each candidate jointly with the question, extra dense "
        "candidates rarely change the final ranking (V4 vs V4h, V5h vs V6 are ties). Why decomposition helped so much: a comparison question "
        "is two lookups whose vocabulary ('compare', 'higher', 'by how much') matches MD&A comparison prose; each single lookup matches the "
        "statement row. Remaining failure modes after G4: wrong-column reads with the right table in context, 'fiscal 2025' answered from a "
        "10-Q's year-to-date column, and occasional self-rounding instead of <calc>.")

    # ------------------------------------------------------------ 10 open items
    D.h("10. Open items and next steps", 1)
    D.bullets([
        ("Ragas (paused): ", "relevancy for all 128 answers with the validated local judge; 120B faithfulness mainly on narrative answers, "
                             "resuming from cache as free quota allows."),
        ("Phase 17: ", "numerical-hallucination evaluation - invented numbers, wrong-cell numbers, and a no-<calc> ablation."),
        ("Phase 18: ", "vector quantization experiment (index size, RAM, latency, recall)."),
        ("Phases 19-21: ", "Phoenix tracing, FastAPI service, Docker packaging."),
        ("Known issues: ", "period ambiguity ('fiscal 2025' without a quarter); lower-confidence sections for Intel and Honeywell 10-Ks."),
    ])

    # ------------------------------------------------------------ appendices
    D.h("Appendix A. Reproduction commands", 1)
    D.code("""
.venv\\Scripts\\python -m src.ingestion.metadata                       # manifest (network: SEC)
.venv\\Scripts\\python -m src.ingestion.downloader                     # download (network: SEC)
.venv\\Scripts\\python -m src.ingestion.verify
.venv\\Scripts\\python -m src.parsing.parser
.venv\\Scripts\\python -m src.chunking.hierarchical
.venv\\Scripts\\python -m src.embeddings.embedder                      # GPU
docker compose up -d                                                   # Qdrant
.venv\\Scripts\\python -m src.retrieval.qdrant_index
.venv\\Scripts\\python -m src.evaluation.dataset --tickers AAPL MSFT NVDA
.venv\\Scripts\\python -m src.evaluation.retrieval --retriever decomposed --version V5   # GPU + LLM (cached)
.venv\\Scripts\\python -m src.evaluation.compare V4_rerank V5_decomposed --metric all_hit@10
.venv\\Scripts\\python -m src.evaluation.generation --version G4 --decompose            # GPU + LLM
.venv\\Scripts\\python -m src.pipeline.scale                                            # GPU
.venv\\Scripts\\python -m src.evaluation.ragas_eval --input G5 G6 --judge groq-gpt-oss-120b   # network, free tier
.venv\\Scripts\\python docs\\tools\\build_report.py                                    # this report
""")
    D.h("Appendix B. Configuration reference", 1)
    ref = [("chunking.text.child_tokens", cfg["chunking"]["text"]["child_tokens"]), ("chunking.text.parent_max_tokens", cfg["chunking"]["text"]["parent_max_tokens"]),
           ("chunking.tables.parent_max_tokens", cfg["chunking"]["tables"]["parent_max_tokens"]), ("embedding.model_name", cfg["embedding"]["model_name"]),
           ("retrieval.bm25.avg_len", cfg["retrieval"]["bm25"]["avg_len"]), ("retrieval.hybrid.alpha", cfg["retrieval"]["hybrid"]["alpha"]),
           ("retrieval.rerank.model", cfg["retrieval"]["rerank"]["model"]), ("retrieval.rerank.default_pool", cfg["retrieval"]["rerank"]["default_pool"]),
           ("retrieval.rerank.batch_size", cfg["retrieval"]["rerank"]["batch_size"]), ("generation.llm.model", cfg["generation"]["llm"]["model"]),
           ("generation.llm.context_window", cfg["generation"]["llm"]["context_window"]), ("generation.context.budget_tokens", cfg["generation"]["context"]["budget_tokens"]),
           ("qdrant.url", cfg["qdrant"]["url"]), ("qdrant.collection", cfg["qdrant"]["collection"]),
           ("evaluation.ragas.default_judge", cfg["evaluation"]["ragas"]["default_judge"])]
    D.table(["Key", "Value"], [(k, str(v)) for k, v in ref], caption="Selected configuration values", widths=[7, 10], numeric_right=False)
    D.source("configs/*.yaml")
    D.h("Appendix C. Glossary", 1)
    D.table(["Term", "Meaning"], [
        ("P@3", "share of the top 3 results that are relevant"), ("R@5 (capped)", "relevant in top 5 / min(relevant, 5)"),
        ("Hit@k", "at least one relevant result in the top k"), ("MRR@10", "mean of 1 / rank of the first relevant result (0 if none in top 10)"),
        ("all-hit@k", "every required fact has a relevant result in the top k"), ("fact_recall@10", "share of required facts found in the top 10"),
        ("BM25", "lexical ranking with term-frequency saturation, length normalisation and IDF"),
        ("Bi-encoder", "encodes query and passage separately into vectors"), ("Cross-encoder", "reads query and passage together and outputs a relevance score"),
        ("RRF", "reciprocal rank fusion: sum of 1/(k + rank) across lists"), ("Relative-score fusion", "min-max-normalised scores mixed by alpha"),
        ("HyDE", "hypothetical document embeddings: search with an LLM-written answer passage"),
        ("Q4_K_M", "4-bit k-quant weight quantization used by llama.cpp/Ollama"), ("KV cache q8_0", "8-bit key/value cache for attention"),
        ("iXBRL", "inline XBRL: machine-readable tags embedded in the filing HTML"), ("McNemar", "paired test on which items flipped between two runs"),
    ], caption="Glossary", widths=[4, 13], numeric_right=False)
    D.h("Appendix D. Run-ID quick reference", 1)
    D.table(["ID", "Meaning"], sorted(((k[0], f"{k[1]}: {k[2]}") for k in key), key=lambda r: r[0].lower()),
            caption="Run IDs (alphabetical)", widths=[3, 14], font=7.5, numeric_right=False)

    D.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
