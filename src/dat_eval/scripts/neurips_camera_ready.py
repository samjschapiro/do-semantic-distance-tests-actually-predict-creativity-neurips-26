"""NeurIPS 2026 paper: Table 1 body, per-temperature table body, the headline figure and the inter-benchmark / inter-test correlation figure.

Every number is computed from the saved scores in data/ (no embeddings or API calls needed):
  data/scores/multi_embed_scores.json   DAT / CDAT / CDAT-N / CDAT-A / PACE per model, per embedding
  data/scores/per_model/<model>/        per-temperature DAT and CDAT scores (appendix table)
  data/scores/cdat_gated_scores.json    appropriateness gate per model and temperature
  data/rat/{pilot_v1,expansion_v1}/summary.json   RAT zero-shot strict accuracy
  data/benchmarks/benchmarks.json       creativity benchmarks and MMLU-Pro
  data/benchmarks/mmlu_pro_raw.json     TIGER-Lab MMLU-Pro scores for models missing above
  data/benchmarks/kombine_composite.json  Kombine overall score

Validity is the Pearson r of a test with a benchmark; specificity is the semi-partial r of the test
with the benchmark residualised on MMLU-Pro (k = 1). The frontier's R for a benchmark is its signed
correlation with MMLU-Pro. Bold / coloured cells in the tables mark p < .05.

Outputs:
  outputs/tables/main_result_body.tex       Table 1 body (tabular only, no caption)
  outputs/tables/per_temperature_body.tex   DAT / CDAT at T = 1.0, 1.5, 2.0 (appendix)
  outputs/figures/fig_headline.pdf/.png
  outputs/figures/fig_benchmark_correlations.pdf/.png
  outputs/cells.json                        every cell with r, p, n

    python -m src.dat_eval.scripts.neurips_camera_ready
"""
import json
import math

import numpy as np
from scipy import stats

from src.dat_eval.scripts.figure_style import (
    C_CAPP, C_CDAT, C_CNOV, C_DAT, C_GREY, C_PACE, _panel_max_ceiling, plt)
from src.utils import ROOT

SCORES = ROOT / "data/scores"
RESULTS = SCORES / "per_model"
BENCH = json.load(open(ROOT / "data/benchmarks/benchmarks.json"))
OUT = ROOT / "outputs"
OUT_FIG = OUT / "figures"
OUT_TAB = OUT / "tables/main_result_body.tex"
OUT_JSON = OUT / "cells.json"

MMLU_RAW_MATCH = {
    "moonshotai_kimi-k2": "Kimi-K2-Instruct",
    "z-ai_glm-4-5-air": "GLM-4.5-Air",
    "google_gemini-3-1-pro-preview": "Gemini-3.1-Pro",
    "google_gemini-3-flash-preview": "Gemini-3-Flash(12/25)",
    "openai_gpt-5-2": "GPT-5.2",
}

TESTS = ["DAT", "CDAT", "CDAT-N", "CDAT-A", "PACE", "RAT"]
TEST_KEY = {"DAT": "dat", "CDAT": "cdat", "CDAT-N": "cdat_novelty",
            "CDAT-A": "cdat_appropriateness", "PACE": "pace"}
TEST_COLORS = {"DAT": C_DAT, "CDAT": C_CDAT, "CDAT-N": C_CNOV, "CDAT-A": C_CAPP,
               "PACE": C_PACE, "RAT": "#8C5A3D"}
CONSTRUCTS = {  # construct -> [(label, key)]
    "Creative Writing": [("Arena CW", "arena_cw"), ("EQ-Bench CW", "eq_bench_cw"),
                         ("Mazur CW", "mazur_cw_v2")],
    "Output Diversity": [("Hivemind Div.", "hivemind_diversity"),
                         ("NovBench Util.", "noveltybench_utility")],
    "Combinatorial Creativity": [("Kombine", "kombine")],
    "Scientific Ideation": [("LiveIdeaBench", "liveideabench")],
}
BENCHMARKS = [b for bs in CONSTRUCTS.values() for b in bs]
EMB_LABEL = {"glove": "GloVe 840B", "fasttext": "FastText crawl-300d-2M",
             "sbert": "Sentence-BERT all-mpnet-base-v2"}


def norm(m: str) -> str:
    return str(m).replace("/", "_").replace(".", "-").replace(":", "_").lower()


def ok(v):
    return v is not None and v not in ("", "---") and not (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------------- data

def load_tests():
    """{block: {test: {model: score}}}; block 'overall' = mean z over the three embeddings."""
    me = json.load(open(SCORES / "multi_embed_scores.json"))
    embs = sorted(me)
    models = sorted({m for e in embs for m in me[e]})
    valid = lambda v: ok(v) and v != 0
    blocks = {e: {t: {norm(m): float(me[e][m][k]) for m in me[e] if valid(me[e][m].get(k))}
                  for t, k in TEST_KEY.items()} for e in embs}
    overall = {}
    for t, k in TEST_KEY.items():
        st = {}
        for e in embs:
            vals = [me[e].get(m, {}).get(k) for m in models]
            vals = [v for v in vals if valid(v)]
            if vals:
                st[e] = (float(np.mean(vals)), float(np.std(vals)) or 1.0)
        col = {}
        for m in models:
            zs = [(me[e][m][k] - st[e][0]) / st[e][1] for e in embs
                  if e in st and valid(me[e].get(m, {}).get(k))]
            if zs:
                col[norm(m)] = float(np.mean(zs))
        overall[t] = col
    rat = {}
    for p in ("expansion_v1", "pilot_v1"):  # pilot wins where a model is in both
        f = ROOT / f"data/rat/{p}/summary.json"
        for m, s in json.load(open(f)).items():
            if s.get("n_total", 0) > 0 and s.get("n_errors", 0) < s["n_total"]:
                rat[norm(m)] = float(s["zs_accuracy_strict"])
    overall["RAT"] = rat
    return {"overall": overall, **blocks}


def load_benchmarks():
    out = {k: {norm(m): float(b[k]) for m, b in BENCH.items() if ok(b.get(k))}
           for _, k in BENCHMARKS if k != "kombine"}
    comp = json.load(open(ROOT / "data/benchmarks/kombine_composite.json"))
    out["kombine"] = {norm(m): float(v["overall"]) for m, v in comp["per_model"].items()
                      if ok(v.get("overall"))}
    for k in ("mmlu_pro", "arena_overall"):
        out[k] = {norm(m): float(b[k]) for m, b in BENCH.items() if ok(b.get(k))}
    raw = json.load(open(ROOT / "data/benchmarks/mmlu_pro_raw.json"))
    for m, row in MMLU_RAW_MATCH.items():
        out["mmlu_pro"].setdefault(m, float(raw[row]))
    return out


# ---------------------------------------------------------------- statistics

def p_of(r, n, k):
    df = n - 2 - k
    if df <= 0 or not -1 < r < 1:
        return float("nan")
    return float(2 * stats.t.sf(abs(r * math.sqrt(df / (1 - r * r))), df))


def cell(test: dict, bench: dict, mmlu: dict) -> dict:
    out = {}
    v_ms = sorted(set(test) & set(bench))
    if len(v_ms) >= 5:
        r = float(stats.pearsonr([test[m] for m in v_ms], [bench[m] for m in v_ms]).statistic)
        out["v"] = {"r": r, "p": p_of(r, len(v_ms), 0), "n": len(v_ms)}
    s_ms = sorted(set(v_ms) & set(mmlu))
    if len(s_ms) >= 5:
        y = np.array([bench[m] for m in s_ms]); g = np.array([mmlu[m] for m in s_ms])
        X = np.column_stack([np.ones(len(g)), g])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        r = float(stats.pearsonr([test[m] for m in s_ms], resid).statistic)
        out["s"] = {"r": r, "p": p_of(r, len(s_ms), 1), "n": len(s_ms)}
    return out


def signed_R(bench: dict, mmlu: dict):
    """Frontier R for k = 1: the benchmark's signed correlation with MMLU-Pro."""
    ms = sorted(set(bench) & set(mmlu))
    if len(ms) < 5:
        return None, len(ms)
    return float(stats.pearsonr([bench[m] for m in ms], [mmlu[m] for m in ms]).statistic), len(ms)


def compute():
    tests, bench = load_tests(), load_benchmarks()
    mmlu = bench["mmlu_pro"]
    cells = {blk: {t: {lab: cell(tests[blk][t], bench[key], mmlu) for lab, key in BENCHMARKS}
                   for t in tests[blk]} for blk in tests}
    R = {lab: dict(zip(("R", "n"), signed_R(bench[key], mmlu))) for lab, key in BENCHMARKS}
    # header n: models with the benchmark and ANY test (validity) / and MMLU-Pro (specificity)
    any_test = set().union(*tests["overall"].values())
    header_n = {lab: (len(set(bench[key]) & any_test), len(set(bench[key]) & any_test & set(mmlu)))
                for lab, key in BENCHMARKS}
    return cells, R, header_n, bench, tests


# ---------------------------------------------------------------- table

def fmt_tex(c, kind, green):
    if kind not in c:
        return "--"
    r, p = c[kind]["r"], c[kind]["p"]
    s = f"{r:+.2f}".replace("0.", ".")
    st = "^{***}" if p < .001 else "^{**}" if p < .01 else "^{*}" if p < .05 else ""
    body = f"${s}{st}$"
    if p < .05:
        body = (r"\good{" if r > 0 else r"\bad{") + body + "}"
    return body


def table_header(header_n):
    """Shared column header (constructs, benchmarks, n, Valid./Spec.) for Table 1 and the
    per-temperature table."""
    nb = len(BENCHMARKS)
    cols = "l" + " c >{\\columncolor{gray!10}}c" * nb
    lines = [r"\begin{tabular}{@{}" + cols + "@{}}", r"\toprule"]
    grp, cm, c0 = [], [], 2
    for cname, bs in CONSTRUCTS.items():
        w = 2 * len(bs)
        # a header wider than its two columns would stretch them; stack it on two lines
        head = (rf"\begin{{tabular}}[b]{{@{{}}c@{{}}}}{cname.replace(' ', r' \\[-1pt] ', 1)}\end{{tabular}}"
                if len(bs) == 1 and len(cname) > 20 else cname)
        grp.append(rf"\multicolumn{{{w}}}{{c}}{{\textit{{{head}}}}}")
        cm.append(rf"\cmidrule(lr){{{c0}-{c0 + w - 1}}}"); c0 += w
    lines += [" & " + "\n & ".join(grp) + r" \\", " " + " ".join(cm)]
    lines.append(" & " + "\n & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{lab} $\uparrow$}}}}"
                                      for lab, _ in BENCHMARKS) + r" \\")
    lines.append(" & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\footnotesize $n{{=}}{header_n[lab][0]}/{header_n[lab][1]}$}}"
        for lab, _ in BENCHMARKS) + r" \\")
    lines.append(" " + " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(nb)))
    lines.append(" & " + " & ".join(["Valid. & Spec."] * nb) + r" \\")
    lines.append(r"\midrule")
    return lines


def write_table(cells, header_n):
    nb = len(BENCHMARKS)
    lines = table_header(header_n)
    # best test per benchmark in the Overall block: highest v + s
    best = {}
    for lab, _ in BENCHMARKS:
        cand = [(c[lab]["v"]["r"] + c[lab]["s"]["r"], t) for t, c in cells["overall"].items()
                if "v" in c[lab] and "s" in c[lab]]
        best[lab] = max(cand)[1] if cand else None
    blocks = [("overall", r"\textbf{Overall (mean $z$-score across 3 embeddings)}")] + \
             [(e, rf"\textit{{{EMB_LABEL[e]}}}") for e in ("glove", "fasttext", "sbert")]
    for bi, (blk, title) in enumerate(blocks):
        if bi:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + 2 * nb}}}{{@{{}}l}}{{{title}}} \\")
        for t in TESTS:
            if t not in cells[blk]:
                continue
            name = rf"\quad {t}" if t in ("CDAT-N", "CDAT-A") else t
            row = [name]
            for lab, _ in BENCHMARKS:
                c = cells[blk][t][lab]
                row += [fmt_tex(c, "v", False), fmt_tex(c, "s", False)]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TAB.parent.mkdir(parents=True, exist_ok=True)
    OUT_TAB.write_text("% generated by src/dat_eval/scripts/neurips_camera_ready.py -- do not edit\n"
                       + "\n".join(lines) + "\n")
    return best




# ---------------------------------------------------------------- per-temperature table (appendix)

TEMPS = ["1.0", "1.5", "2.0"]
MIN_VALID = 0.70  # share of trials / cues that must parse at a temperature for the score to count
OUT_TEMP_TAB = OUT / "tables/per_temperature_body.tex"


def load_per_temperature():
    """{(test, T): {model: score}} for DAT (GloVe) and gated CDAT (SBERT) at each temperature.

    Per-temperature scores are stored only under each test's primary embedding, so this table uses
    GloVe for DAT and SBERT for CDAT rather than Table 1's three-embedding composite. CDAT at T is the
    novelty at T, kept only if the appropriateness gate (Welch t vs random nouns, BH-FDR across
    models, adjusted p < .001) passed at that same T.
    """
    gate = json.load(open(SCORES / "cdat_gated_scores.json"))["sbert"]
    out = {}
    for md in sorted(RESULTS.iterdir()):
        if not md.is_dir():
            continue
        m = norm(md.name)
        f = md / "dat_scores.json"
        if f.exists():
            for T, d in (json.load(open(f)).get("by_temperature") or {}).items():
                n, ns = d.get("n_trials") or 0, d.get("n_sufficient") or 0
                if T in TEMPS and ok(d.get("mean_score")) and n and ns / n >= MIN_VALID:
                    out.setdefault(("DAT", T), {})[m] = float(d["mean_score"])
        f = md / "cdat_scores.json"
        g = (gate.get(md.name) or {}).get("gate") or {}
        if f.exists():
            for T, d in (json.load(open(f)).get("by_temperature") or {}).items():
                n, ns = d.get("n_cues") or 0, d.get("n_sufficient") or 0
                if (T in TEMPS and (g.get(T) or {}).get("passed") and ok(d.get("mean_novelty"))
                        and n and ns / n >= MIN_VALID):
                    out.setdefault(("CDAT", T), {})[m] = float(d["mean_novelty"])
    return out


def write_per_temperature(header_n_unused=None):
    bench = load_benchmarks(); mmlu = bench["mmlu_pro"]
    pt = load_per_temperature()
    cells = {f"{t} T={T}": {lab: cell(pt[(t, T)], bench[key], mmlu) for lab, key in BENCHMARKS}
             for (t, T) in sorted(pt, key=lambda k: (k[0] != "DAT", k[1]))}
    any_m = set().union(*pt.values())
    hn = {lab: (len(set(bench[key]) & any_m), len(set(bench[key]) & any_m & set(mmlu)))
          for lab, key in BENCHMARKS}
    nb = len(BENCHMARKS)
    lines = table_header(hn)
    for bi, t in enumerate(("DAT", "CDAT")):
        if bi:
            lines.append(r"\midrule")
        emb = "GloVe 840B" if t == "DAT" else "Sentence-BERT all-mpnet-base-v2"
        lines.append(rf"\multicolumn{{{1 + 2 * nb}}}{{@{{}}l}}{{\textbf{{{t}}} \textit{{({emb})}}}} \\")
        for T in TEMPS:
            c = cells[f"{t} T={T}"]
            row = [rf"\quad $T = {T}$"]
            for lab, _ in BENCHMARKS:
                row += [fmt_tex(c[lab], "v", False), fmt_tex(c[lab], "s", False)]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TEMP_TAB.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEMP_TAB.write_text("% generated by src/dat_eval/scripts/neurips_camera_ready.py -- do not edit\n"
                            + "\n".join(lines) + "\n")
    # rank-order stability across temperatures (models present at both temperatures)
    stab = {}
    for t in ("DAT", "CDAT"):
        for a, b in (("1.0", "1.5"), ("1.0", "2.0"), ("1.5", "2.0")):
            ms = sorted(set(pt[(t, a)]) & set(pt[(t, b)]))
            r = float(stats.pearsonr([pt[(t, a)][m] for m in ms], [pt[(t, b)][m] for m in ms]).statistic)
            stab[f"{t} {a} vs {b}"] = {"r": r, "p": p_of(r, len(ms), 0), "n": len(ms)}
    return cells, hn, stab



# ---------------------------------------------------------------- headline figure

def _place_labels(ax, pts, obstacles, xr, yr, fs, force=None):
    """Greedy label placement next to each construct-mean point: try 8 directions at a fixed
    offset in points, keep the one whose label box is farthest from every dot (means and cells)
    and from labels already placed. Distances are in axis-fraction units."""
    cands = [(1, 0, "left", "center"), (-1, 0, "right", "center"), (0, 1, "center", "bottom"),
             (0, -1, "center", "top"), (1, 1, "left", "bottom"), (-1, 1, "right", "bottom"),
             (1, -1, "left", "top"), (-1, -1, "right", "top")]
    fx = lambda x: (x - xr[0]) / (xr[1] - xr[0])
    fy = lambda y: (y - yr[0]) / (yr[1] - yr[0])
    off = 13 * fs / 15  # points, grows with the label font
    w_char, h = 0.015 * fs / 15, 0.055 * fs / 15  # rough label size in axis fraction (bold, generous)
    dots = [(fx(x), fy(y)) for x, y in obstacles]
    placed = []
    for name, (x, y) in pts.items():
        px, py = fx(x), fy(y)
        w = w_char * len(name)
        best, best_score = None, -1e9
        pool = [c for c in cands if (c[0], c[1]) == force[name]] if force and name in force else cands
        for (dx0, dy0, ha, va), far in [(c, 1.0) for c in pool]:
            dx, dy = dx0 * far, dy0 * far  # also try twice the offset when the near spots are crowded
            ox, oy = px + dx * 0.022 * fs / 15, py + dy * 0.035 * fs / 15
            cx = ox + (w / 2 if ha == "left" else -w / 2 if ha == "right" else 0)
            cy = oy + (h / 2 if va == "bottom" else -h / 2 if va == "top" else 0)
            def gap(qx, qy):  # distance from a point to the label box
                return math.hypot(max(abs(qx - cx) - w / 2, 0), max(abs(qy - cy) - h / 2, 0))
            d_dots = min([gap(qx, qy) for qx, qy in dots if (qx, qy) != (px, py)] or [1])
            d_lab = min([math.hypot(max(abs(cx - lx) - (w + lw) / 2, 0), max(abs(cy - ly) - h, 0))
                         for lx, ly, lw in placed] or [1])
            inside = 0.01 < cx - w / 2 and cx + w / 2 < 0.99 and 0.01 < cy - h / 2 and cy + h / 2 < 0.99
            d_own = gap(px, py)
            others = [gap(qx, qy) for qx, qy in dots if (qx, qy) != (px, py)]
            wrong = any(o < d_own for o in others)  # label would sit nearer another dot: misleading
            score = (min(d_dots, d_lab) - (0 if inside else 1) - (0.5 if wrong else 0)
                     - 0.002 * (abs(dx0) + abs(dy0)) - (0.02 if far > 1 else 0))
            if score > best_score:
                best, best_score = (dx, dy, ha, va, cx, cy), score
        dx, dy, ha, va, cx, cy = best
        placed.append((cx, cy, w))
        n = math.hypot(dx, dy) / max(abs(dx), abs(dy), 1)  # keep the far factor in the offset
        ax.annotate(name, xy=(x, y), xytext=(dx / n * off, dy / n * off), textcoords="offset points",
                    ha=ha, va=va, fontsize=fs, fontweight="bold", color="black", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))


# Figure 6 is drawn at 22 in and printed at ~5.5 in; HFS scales every font so text reads at ~7-9 pt
HFS = 1.45


def fig_headline(cells, R):
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as mtransforms

    ov = cells["overall"]
    v_grid = np.linspace(-1, 1, 400)

    def pts_of(labels, t):
        out = []
        for lab in labels:
            c = ov[t][lab]
            if "v" in c and "s" in c:
                out.append((c["v"]["r"], c["s"]["r"], c["v"]["p"] < .05, c["s"]["p"] < .05))
        return out

    def ceiling(Rs):
        return _panel_max_ceiling([r for r in Rs if r is not None], v_grid)

    fig = plt.figure(figsize=(22.0, 13.9))  # taller for the (a)-(b) gap and a two-row legend; panel sizes unchanged
    nc = len(CONSTRUCTS)
    gs_top = GridSpec(1, nc, figure=fig, left=0.045, right=0.99, top=0.9204, bottom=0.5997, wspace=0.12)
    gs_bot = GridSpec(1, len(BENCHMARKS), figure=fig, left=0.045, right=0.99, top=0.4042,
                      bottom=0.19, wspace=0.26)
    ax_top = [fig.add_subplot(gs_top[0, 0])]
    ax_top += [fig.add_subplot(gs_top[0, i], sharex=ax_top[0], sharey=ax_top[0]) for i in range(1, nc)]
    axes_bot = [fig.add_subplot(gs_bot[0, 0])]
    axes_bot += [fig.add_subplot(gs_bot[0, i], sharex=axes_bot[0], sharey=axes_bot[0])
                 for i in range(1, len(BENCHMARKS))]
    fig.text(0.5, 0.9776, "(a) Prediction by construct", ha="center", va="bottom",
             fontsize=28.0 * HFS, fontweight="bold")
    fig.text(0.5, 0.4491, "(b) Prediction by benchmark", ha="center", va="bottom",
             fontsize=28.0 * HFS, fontweight="bold")

    # shared limits for the top row
    allx, ally = [], []
    for cname, bs in CONSTRUCTS.items():
        labs = [b for b, _ in bs]
        for t in TESTS:
            p = pts_of(labs, t)
            allx += [q[0] for q in p] + ([np.mean([q[0] for q in p])] if p else [])
            ally += [q[1] for q in p] + ([np.mean([q[1] for q in p])] if p else [])
        cy = ceiling([R[b]["R"] for b in labs]); j = int(np.argmax(v_grid + cy))
        allx.append(v_grid[j]); ally.append(cy[j] + 0.06)
    xpad, ypad = 0.12 * (max(allx) - min(allx)), 0.10 * (max(ally) - min(ally))
    xr, yr = (min(allx) - xpad, max(allx) + xpad), (min(ally) - ypad, max(ally) + ypad)

    for ax, (cname, bs) in zip(ax_top, CONSTRUCTS.items()):
        labs = [b for b, _ in bs]
        ax.add_patch(Rectangle((0, 0), 10, 10, facecolor="#7ec587", alpha=0.10,
                               edgecolor="none", zorder=-1))
        ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        ax.axvline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        cy = ceiling([R[b]["R"] for b in labs])
        ax.plot(v_grid, cy, color="black", linewidth=1.6, alpha=0.9, zorder=1)
        j = int(np.argmax(v_grid + cy))
        ax.scatter(v_grid[j], cy[j], marker="D", s=70, facecolor="black", edgecolor="black",
                   linewidth=0.8, zorder=7)
        ax.text(0.98, 0.03, f"$(v^*, s^*)=({v_grid[j]:+.2f}, {cy[j]:+.2f})$", transform=ax.transAxes,
                fontsize=17.0 * HFS, ha="right", va="bottom", zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.85))
        star_trans = mtransforms.offset_copy(ax.transData, fig=fig, x=8, y=8, units="points")
        means, obstacles = {}, []
        for t in TESTS:
            p = pts_of(labs, t)
            if not p:
                continue
            for v, s_, vs, ss in p:
                obstacles.append((v, s_))
                anys = vs or ss
                ax.scatter(v, s_, marker="o", s=60, c=[TEST_COLORS[t]],
                           edgecolor=("black" if anys else "white"),
                           linewidth=(1.0 if anys else 0.4), zorder=3, alpha=0.55)
                if vs and ss and v > 0 and s_ > 0:
                    ax.scatter(v, s_, marker="*", s=65, facecolor="#ffcc00", edgecolor="black",
                               linewidth=0.4, zorder=5, transform=star_trans)
            mx, my = float(np.mean([q[0] for q in p])), float(np.mean([q[1] for q in p]))
            ax.scatter(mx, my, marker="o", s=290, c=[TEST_COLORS[t]], edgecolor="black",
                       linewidth=1.4, zorder=4)
            means[t] = (mx, my); obstacles.append((mx, my))
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        _place_labels(ax, means, obstacles, xr, yr, 15.0 * 1.25,
                      force={"CDAT": (1, -1)} if cname == "Creative Writing" else None)
        ax.set_title(cname, fontsize=23.0 * HFS)
        ax.tick_params(axis="both", labelsize=17.0 * HFS)
        ax.set_xlabel(r"Validity  ($r$)", fontsize=21.0 * HFS)
    ax_top[0].set_ylabel(r"Specificity  ($r \mid g$)", fontsize=21.0 * HFS)
    for ax in ax_top[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    for ax, (lab, _) in zip(axes_bot, BENCHMARKS):
        c = R[lab]["R"]
        r1c = math.sqrt(max(0.0, 1 - c ** 2)); r1v = np.sqrt(np.clip(1 - v_grid ** 2, 0, None))
        upper, lower = v_grid * r1c + abs(c) * r1v, v_grid * r1c - abs(c) * r1v
        ax.fill_between(v_grid, lower, upper, color="#d0d0d0", alpha=0.35, zorder=0)
        ax.plot(v_grid, upper, color="black", linewidth=1.6, zorder=1)
        ax.plot(v_grid, lower, color="black", linewidth=0.7, linestyle=":", alpha=0.5, zorder=1)
        ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        ax.axvline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        j = int(np.argmax(v_grid + upper))
        ax.scatter(v_grid[j], upper[j], marker="D", s=55, facecolor="black", edgecolor="black",
                   linewidth=0.8, zorder=7)
        for t in TESTS:
            cc = ov[t][lab]
            if "v" in cc and "s" in cc:
                ax.scatter(cc["v"]["r"], cc["s"]["r"], marker="o", s=85, c=[TEST_COLORS[t]],
                           edgecolor="black", linewidth=0.9, zorder=3, alpha=0.95)
        ax.set_title(lab, fontsize=20.0 * HFS)
        ax.tick_params(axis="both", labelsize=16.0 * HFS)
        ax.set_xlabel(r"Validity  ($r$)", fontsize=20.0 * HFS)
    axes_bot[0].set_xlim(-1.02, 1.02); axes_bot[0].set_ylim(-1.05, 1.05)
    axes_bot[0].set_ylabel(r"Specificity  ($r \mid g$)", fontsize=21.0 * HFS)
    for ax in axes_bot[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=TEST_COLORS[t],
                      markeredgecolor="black", markeredgewidth=1.0, markersize=15, label=t)
               for t in TESTS]
    handles += [Line2D([], [], color="black", linewidth=1.4, label="theoretical ceiling"),
                Line2D([], [], marker="D", linestyle="none", markerfacecolor="black",
                       markeredgecolor="black", markersize=10,
                       label=r"$(v^*, s^*) = \arg\max\,(v + s)$")]
    ncol = (len(handles) + 1) // 2  # two rows; matplotlib fills columns first, so interleave to read across
    rows = [handles[:ncol], handles[ncol:]]
    handles = [h for pair in zip(*rows) for h in pair] + rows[0][len(rows[1]):]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=ncol,
               frameon=False, fontsize=22 * HFS, handletextpad=0.4, columnspacing=2.0)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    out = OUT_FIG / "fig_headline.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png")); plt.close()
    return out


# ---------------------------------------------------------------- Figure 3: significant cells only

def _sig_triangle(ax, mat, pmat, labels, fs=14, cell_fs=22):
    """Lower triangle coloured by r for every cell; the value is printed (bold, cell_fs) only where p < .05."""
    from matplotlib.patches import Rectangle
    n = len(labels)
    disp = np.ma.masked_array(np.nan_to_num(mat), mask=~np.tri(n, dtype=bool) | np.isnan(mat))
    im = ax.imshow(disp, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    for i in range(n):
        for j in range(i + 1):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black",
                                   linewidth=0.2, zorder=4))
            v = mat[i, j]
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=cell_fs, fontweight="bold", color="white", zorder=3)
            elif not np.isnan(v) and pmat[i, j] < .05:  # value printed only when significant
                ax.text(j, i, f"{v:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=cell_fs, fontweight="bold",
                        color=("white" if abs(v) > 0.6 else "black"), zorder=3)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.tick_params(axis="both", which="major", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def _corr_mats(series: list[dict]):
    n = len(series)
    mat, pm, nm = np.full((n, n), np.nan), np.full((n, n), np.nan), np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            ms = sorted(set(series[i]) & set(series[j]))
            nm[i, j] = len(ms)
            if len(ms) >= 5:
                r, p = stats.pearsonr([series[i][m] for m in ms], [series[j][m] for m in ms])
                mat[i, j], pm[i, j] = r, p
    return mat, pm, nm


def fig_benchmark_correlations(bench, tests):
    from matplotlib.gridspec import GridSpec
    keys_a = ["mmlu_pro", "arena_cw", "eq_bench_cw", "mazur_cw_v2",
              "hivemind_diversity", "noveltybench_utility", "kombine", "liveideabench"]
    labels_a = ["MMLU-Pro", "Arena CW", "EqBench CW", "Mazur V2", "HiveMind",
                "NoveltyBench", "Kombine", "LiveIdea"]
    mat_a, p_a, n_a = _corr_mats([bench[k] for k in keys_a])
    mat_b, p_b, n_b = _corr_mats([tests["overall"][t] for t in TESTS])
    na, nb = len(keys_a), len(TESTS)
    fig = plt.figure(figsize=(16.0, 7.0))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[na, nb], wspace=0.20)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    im = _sig_triangle(ax1, mat_a, p_a, labels_a)
    _sig_triangle(ax2, mat_b, p_b, TESTS)
    fig.canvas.draw()
    pos1 = ax1.get_position(); cs = pos1.height / na; pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, pos1.y0, pos2.width, cs * nb])
    ax1.set_title("(a) Inter-benchmark correlations", fontsize=19, loc="left", pad=6)
    ax2.set_title("(b) Inter-test correlations", fontsize=19, loc="left", pad=6)
    cb = fig.colorbar(im, ax=[ax1, ax2], shrink=0.7, pad=0.03)
    cb.set_label("Pearson $r$", fontsize=14); cb.ax.tick_params(labelsize=12)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    out = OUT_FIG / "fig_benchmark_correlations.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    return out, {"benchmarks": {"labels": labels_a, "r": mat_a.tolist(), "p": p_a.tolist(), "n": n_a.tolist()},
                 "tests": {"labels": TESTS, "r": mat_b.tolist(), "p": p_b.tolist(), "n": n_b.tolist()}}


# ---------------------------------------------------------------- main

def main():
    cells, R, header_n, bench, tests = compute()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    best = write_table(cells, header_n)
    temp_cells, temp_n, temp_stab = write_per_temperature()
    print("per-temperature n:", temp_n)
    for k, v in temp_stab.items():
        print(f"  stability {k}: r={v['r']:+.2f} p={v['p']:.1e} n={v['n']}")
    f1 = fig_headline(cells, R)
    f3, fig3 = fig_benchmark_correlations(bench, tests)
    OUT_JSON.write_text(json.dumps({"cells": cells, "R_mmlu": R, "header_n": header_n,
                                    "best_overall": best, "fig3": fig3,
                                    "per_temperature": {"cells": temp_cells, "header_n": temp_n,
                                                        "stability": temp_stab}}, indent=1))
    print(f"wrote {f1}, {f3}")
    print("frontier R (signed corr with MMLU-Pro):",
          {k: (round(v["R"], 2) if v["R"] is not None else None, v["n"]) for k, v in R.items()})
    print("header n:", header_n)
    print("best:", best)
    for t in TESTS:
        row = f"{t:7s}"
        for lab, _ in BENCHMARKS:
            c = cells["overall"][t][lab]
            f = lambda k: (f"{c[k]['r']:+.2f}{'*' if c[k]['p'] < .05 else ' '}" if k in c else "  -- ")
            row += f" | {f('v')} {f('s')}"
        print(row)
    print(f"wrote {OUT_TAB}, {OUT_JSON}")


if __name__ == "__main__":
    main()
