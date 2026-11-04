"""Build the progress-report deck: python presentation/build_deck.py

Requires python-pptx. Week-2 measurements are placeholders marked "[fill in]";
either edit WEEK2 below and rebuild, or edit the .pptx directly.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).parent
OUT = HERE / "progress_report_week2.pptx"

# Week-2 numbers from `agent-eval bench` and the pilot trajectories. Replace None with values.
WEEK2 = {
    "tok_s_1.7b": None,
    "tok_s_4b": None,
    "tok_s_8b": None,
    "gpu_hours_projection": None,
    "pilot_parse_fail": None,
    "pilot_finish_rate": None,
    "pilot_budget_hit": None,
}

NAVY = RGBColor(0x1E, 0x29, 0x3B)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
TEAL_LIGHT = RGBColor(0xCC, 0xEB, 0xE7)
AMBER = RGBColor(0xB4, 0x53, 0x09)
AMBER_LIGHT = RGBColor(0xFE, 0xF3, 0xC7)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
BLUE_LIGHT = RGBColor(0xDB, 0xEA, 0xFE)
RED = RGBColor(0xB9, 0x1C, 0x1C)
RED_LIGHT = RGBColor(0xFE, 0xE2, 0xE2)
GREEN = RGBColor(0x15, 0x80, 0x3D)
GREEN_LIGHT = RGBColor(0xDC, 0xFC, 0xE7)
GREY = RGBColor(0x64, 0x74, 0x8B)
GREY_LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
LINE = RGBColor(0xCB, 0xD5, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

N_CONTENT = 12


def fill_in(key: str, fmt: str = "{}") -> str:
    v = WEEK2.get(key)
    return "[fill in]" if v is None else fmt.format(v)


# ------------------------------------------------------------------ primitives


def _runs(paragraph, text: str, size: float, color: RGBColor, bold: bool = False, italic: bool = False):
    """Add text with **bold** spans to a paragraph."""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = paragraph.add_run()
        r.text = part
        f = r.font
        f.name = FONT
        f.size = Pt(size)
        f.bold = bold or (i % 2 == 1)
        f.italic = italic
        f.color.rgb = AMBER if part == "[fill in]" else color


def text(slide, x, y, w, h, paras, size=16, color=NAVY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         bold=False, spacing=4, bullet=False, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    if isinstance(paras, str):
        paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(spacing)
        psize, pcolor, pbold = size, color, bold
        if isinstance(p, tuple):
            p, opts = p
            psize = opts.get("size", size)
            pcolor = opts.get("color", color)
            pbold = opts.get("bold", bold)
        _runs(para, ("•  " + p) if bullet else p, psize, pcolor, pbold, italic)
    return tb


def box(slide, x, y, w, h, fill=GREY_LIGHT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.text = ""
    return s


def card(slide, x, y, w, h, title, body, accent=TEAL, fill=GREY_LIGHT, title_size=17, body_size=14, bullet=True):
    box(slide, x, y, w, h, fill=fill)
    box(slide, x, y, 0.09, h, fill=accent, shape=MSO_SHAPE.RECTANGLE)
    text(slide, x + 0.22, y + 0.12, w - 0.35, 0.45, title, size=title_size, color=accent, bold=True)
    text(slide, x + 0.22, y + 0.6, w - 0.35, h - 0.7, body, size=body_size, bullet=bullet, spacing=5)


def arrow(slide, x, y, w=0.4, h=0.32, color=GREY):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False


def table(slide, x, y, w, rows, col_w, size=13, header_fill=NAVY, row_h=0.42, first_col_bold=True, zebra=True):
    n_r, n_c = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(row_h * n_r))
    tbl = shape.table
    tbl.first_row = True
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
            elif zebra and i % 2 == 0:
                cell.fill.fore_color.rgb = GREY_LIGHT
            else:
                cell.fill.fore_color.rgb = WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            para = tf.paragraphs[0]
            if i == 0:
                _runs(para, val, size, WHITE, bold=True)
            else:
                _runs(para, val, size, NAVY, bold=(j == 0 and first_col_bold))
    return tbl


def base(prs, kicker: str, title: str, n: int):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box(s, 0, 0, 13.333, 0.12, fill=TEAL, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.6, 0.32, 8, 0.35, kicker.upper(), size=12, color=TEAL, bold=True)
    text(s, 0.6, 0.62, 12.2, 0.75, title, size=28, color=NAVY, bold=True)
    s.shapes.add_connector(1, Inches(0.6), Inches(7.0), Inches(12.73), Inches(7.0)).line.color.rgb = LINE
    text(s, 0.6, 7.05, 8, 0.3, "CS 6180 · Does the Final Answer Measure the Agent? · Progress report", size=10, color=GREY)
    text(s, 11.2, 7.05, 1.53, 0.3, f"{n} / {N_CONTENT}", size=10, color=GREY, align=PP_ALIGN.RIGHT)
    return s


def notes(slide, body: str):
    slide.notes_slide.notes_text_frame.text = body.strip()


# ---------------------------------------------------------------------- slides


def title_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box(s, 0, 0, 13.333, 7.5, fill=NAVY, shape=MSO_SHAPE.RECTANGLE)
    box(s, 0.8, 2.05, 0.12, 2.1, fill=TEAL, shape=MSO_SHAPE.RECTANGLE)
    text(s, 1.15, 1.95, 11.9, 1.0, "Does the Final Answer Measure the Agent?", size=38, color=WHITE, bold=True)
    text(s, 1.15, 3.0, 11, 1.2,
         "Outcome-only vs. trajectory-aware vs. human evaluation of a research-assistant agent",
         size=24, color=RGBColor(0xCB, 0xD5, 0xE1))
    text(s, 1.15, 4.6, 11, 0.5, "Progress report · weeks 1–2 of 14 complete", size=20, color=RGBColor(0x5E, 0xEA, 0xD4), bold=True)
    text(s, 1.15, 5.6, 11, 0.9,
         ["[Student Name]", "CS 6180 Foundations of Generative AI · Fall 2026"],
         size=16, color=RGBColor(0xCB, 0xD5, 0xE1))
    notes(s, """
[~15 s]
Hi everyone. My project asks a simple-sounding question: when we evaluate an LLM agent, does looking at the final answer actually tell us how good the agent is?
I'll cover why this matters and what prior work did, my research question and hypotheses, the approach, what I've finished in the first two weeks with the first findings, and the plan for the rest of the semester.
""")


def s_background(prs, n):
    s = base(prs, "Motivation", "Agents are graded on their last line", n)
    # Agent loop strip
    steps = ["Question", "search", "read", "reason", "Final answer"]
    x = 0.6
    for i, st in enumerate(steps):
        last = i == len(steps) - 1
        w = 2.05 if last else 1.55
        b = box(s, x, 1.6, w, 0.62, fill=AMBER_LIGHT if last else TEAL_LIGHT, line=AMBER if last else None)
        text(s, x, 1.6, w, 0.62, st, size=16, color=AMBER if last else TEAL, bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        x += w
        if not last:
            arrow(s, x + 0.08, 1.75, 0.38, 0.32)
            x += 0.55
    text(s, 9.05, 2.3, 3.7, 0.4, "↑ usually the only thing scored", size=14, color=AMBER, bold=True)
    text(s, 0.6, 2.3, 7.5, 0.4, "The research trajectory: several tool calls, all hidden from the score", size=14, color=GREY, italic=True)

    card(s, 0.6, 3.05, 5.9, 2.35, "Right answer, bad process",
         ["Recalls the answer from pretraining instead of retrieving it",
          "Cites a paragraph that does not support the answer",
          "Gets lucky on 1 of 3 runs"],
         accent=AMBER, fill=AMBER_LIGHT, body_size=15)
    text(s, 5.35, 3.12, 1.0, 0.5, "✓", size=26, color=GREEN, bold=True, align=PP_ALIGN.RIGHT)
    card(s, 6.83, 3.05, 5.9, 2.35, "Wrong answer, good process",
         ["Finds both supporting paragraphs …",
          "… then picks the wrong bridge entity",
          "Or recovers well from a tool error but runs out of steps"],
         accent=BLUE, fill=BLUE_LIGHT, body_size=15)
    text(s, 11.6, 3.12, 1.0, 0.5, "✗", size=26, color=RED, bold=True, align=PP_ALIGN.RIGHT)

    box(s, 0.6, 5.65, 12.13, 1.1, fill=GREY_LIGHT)
    text(s, 0.85, 5.72, 11.7, 1.0,
         ["**Why it matters:** leaderboards rank agents by final-answer accuracy, RL methods such as Search-R1 "
          "train agents on an exact-match reward alone, and deployment decisions follow those numbers. "
          "If the score hides process failures, we optimize and ship the wrong thing."],
         size=15, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, """
[~60 s]
LLM agents like research assistants work in several steps: they search, read documents, reason, and then answer. But most evaluations only check whether that final answer matches a reference.
That hides two kinds of cases. On the left, the agent gets the right answer through a bad process: maybe it already memorized the answer during pretraining and the search was decoration, maybe it cites a document that doesn't actually support the answer, or it just got lucky on one sample. Outcome-only evaluation gives it full credit.
On the right, the agent does the research well, finds both pieces of evidence, and then makes one reasoning slip. Outcome-only evaluation gives it zero.
This matters because leaderboards, and even training methods like Search-R1 that use exact match as the only reward, are built on that final-answer number. If the number is misleading, we optimize and deploy the wrong agents.
""")


def s_prior(prs, n):
    s = base(prs, "Background", "Prior work: outcome-only is not enough, but by how much?", n)
    rows = [
        ["Work", "What it shows", "What is still missing"],
        ["AI Agents That Matter · HAL (Kapoor et al. 2024–26)", "Cost-controlled, Pareto evaluation; more reasoning effort often lowers accuracy", "Quality is still task success; logs are read qualitatively"],
        ["TRACE (2026)", "\"High-score illusion\" of Pass@1; scores efficiency and evidence grounding", "Evaluator not validated against humans, bias, or repeated runs"],
        ["AgentProcessBench (2026)", "8,509 step labels (+1/0/−1); weak agents look good by stopping early", "Benchmarks judges, not outcome-vs-process disagreement"],
        ["TRAIL · MAST (2025)", "Error taxonomies for agent traces; κ = 0.88 with six experts", "~110 min per trace; not linked to outcome scores"],
        ["Judging LLM-as-a-Judge (Zheng et al. 2023)", "Position, verbosity and self-preference bias in LLM judges", "Single-turn chat, not agent trajectories"],
        ["τ-bench (Yao et al. 2024)", "pass^k: GPT-4o drops from >60% to <25% at k = 8", "Shows that runs diverge, not why"],
    ]
    table(s, 0.6, 1.5, 12.13, rows, [3.35, 4.6, 4.18], size=13, row_h=0.56)
    box(s, 0.6, 5.65, 12.13, 1.1, fill=TEAL_LIGHT)
    text(s, 0.85, 5.7, 11.7, 1.0,
         "**Gap:** no study scores the **same trajectories** with outcome, trajectory and human evaluation, and measures "
         "**how often** they disagree, **whether rankings change** at matched cost, and **how reliable** each evaluator is.",
         size=16, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, """
[~60 s]
I'm not the first to say outcome-only evaluation is insufficient. Kapoor and colleagues argue for cost-controlled, Pareto evaluation. TRACE, the closest work, calls it the "high-score illusion" and scores trajectories, but doesn't validate that evaluator against humans or repeated runs.
AgentProcessBench shows weak agents can look good by stopping early. TRAIL and MAST built error taxonomies, but they're expensive and not linked to outcome scores.
Zheng et al. showed LLM judges are biased, but on chat, not trajectories. And tau-bench's pass^k shows agents are unreliable across runs, without explaining why.
The gap: nobody scores the same trajectories all three ways and measures the disagreement, its effect on rankings, and each evaluator's reliability.
""")


def s_rq(prs, n):
    s = base(prs, "Research question", "The question this project answers", n)
    box(s, 0.6, 1.5, 12.13, 1.75, fill=NAVY)
    text(s, 1.0, 1.55, 11.4, 1.65,
         "On identical agent trajectories, how much do outcome-only and trajectory-aware evaluations disagree, "
         "does it change which configuration is judged best, and which automatic evaluator best tracks "
         "human judgment per unit of cost?",
         size=21, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    items = [
        ("RQ1 · Disagreement", "How often is the answer right but the process bad, or the reverse?", AMBER, AMBER_LIGHT),
        ("RQ2 · Rankings", "Do outcome, trajectory and human scores rank the agent configurations the same way at matched cost?", BLUE, BLUE_LIGHT),
        ("RQ3 · Validity", "Which automatic metric best predicts human judgment, and what does it cost to compute?", TEAL, TEAL_LIGHT),
        ("RQ4 · Reliability", "How much does a single run overstate reliability across repeated runs (pass^k)?", RED, RED_LIGHT),
    ]
    x = 0.6
    for t, d, c, f in items:
        card(s, x, 3.55, 2.88, 2.55, t, [d], accent=c, fill=f, title_size=16, body_size=15, bullet=False)
        x += 3.08
    text(s, 0.6, 6.3, 12.1, 0.5,
         "Optional RQ5 (only if the week-9 gate passes): under injected tool errors, do trajectory metrics catch degradation that outcome scores miss?",
         size=13, color=GREY, italic=True)
    notes(s, """
[~45 s]
Here's the research question in one sentence: on identical trajectories, how much do outcome-only and trajectory-aware evaluations disagree, does that change which configuration we'd call best, and which automatic evaluator tracks human judgment best for its cost.
It breaks into four parts. RQ1 is the disagreement itself. RQ2 asks whether it changes decisions, meaning rankings of configurations at the same token cost. RQ3 asks which cheap automatic signal is the best proxy for a human. RQ4 asks how much a single run overstates reliability.
There is an optional fifth question about robustness to tool errors, which I'll only run if I'm on schedule at week 9.
""")


def s_hyp(prs, n):
    s = base(prs, "Hypotheses", "Pre-registered predictions with explicit decision rules", n)
    rows = [
        ["", "Prediction", "Measured by", "Supported if"],
        ["H1a", "At least 10% of correct answers are **ungrounded** (right answer, bad process)", "Correct × grounded 2×2 per config", "Rate ≥ 10% in every ReAct config"],
        ["H1a-mech", "Those answers come from **memorization**", "Closed-book accuracy on the same questions", "Gap > 0, 95% CI excludes 0"],
        ["H1b", "The rate **shrinks with model size**", "Rate for 1.7B vs 4B vs 8B", "Strictly decreasing"],
        ["H2", "Human and outcome **rankings differ** at matched cost", "Kendall τ over the 6 ReAct configs", "τ < 0.8"],
        ["H3a", "Cheap rule-based metrics predict humans **as well as an LLM judge**", "Spearman ρ with the human score", "Best rule ρ ≥ LLM-judge ρ"],
        ["H3b", "LLM judges are **measurably biased** on trajectories", "Position swap, filler padding, 2nd-family judge", "Any of the three tests flags bias"],
        ["H4", "One run **overstates reliability**", "pass^3 ÷ pass^1", "< 0.7 for every config"],
    ]
    table(s, 0.6, 1.45, 12.13, rows, [1.25, 4.7, 3.38, 2.8], size=13, row_h=0.56)
    text(s, 0.6, 6.05, 12.13, 0.85,
         ["**Grounded** = cites at least one gold paragraph and only gold paragraphs.  "
          "**Pre-registration (week 3):** hypotheses, metrics, question IDs and decision rules are frozen and tagged in git "
          "before the main runs. All are low-confidence guesses; any that turn out wrong are reported as wrong."],
         size=13, color=GREY)
    notes(s, """
[~75 s]
Each hypothesis has a measurement and a decision rule written down in advance.
H1a: at least 10% of correct answers are ungrounded, meaning the agent didn't cite the gold evidence. H1a-mech says why: memorization, tested against the same model answering closed-book. H1b predicts the problem shrinks with model size.
H2: at matched token budget, ranking by human judgment differs from ranking by accuracy, Kendall tau below 0.8.
H3a: cheap rule-based metrics predict human judgment as well as an LLM judge. H3b: the LLM judge shows position, verbosity or self-preference bias on trajectories.
H4: passing all three runs is less than 70% as likely as passing one.
These are deliberately low-confidence guesses. They're frozen in week 3, and if they're wrong, the paper says so.
""")


def s_system(prs, n):
    s = base(prs, "Approach", "One agent, a frozen corpus, a controlled grid", n)
    blocks = [
        ("Frozen corpus", "66,581 HotpotQA paragraphs, each question's gold evidence inside", TEAL_LIGHT, TEAL),
        ("Tools", "search (BM25, top-5) · read · finish(answer, citations)", TEAL_LIGHT, TEAL),
        ("ReAct agent", "Qwen3 1.7B · 4B · 8B, thinking on/off (1,024-token budget), ≤ 10 steps", BLUE_LIGHT, BLUE),
        ("Trajectory log", "Every thought, action, observation, token count → JSONL", AMBER_LIGHT, AMBER),
    ]
    x, w = 0.6, 2.72
    for i, (t, d, f, c) in enumerate(blocks):
        box(s, x, 1.55, w, 1.65, fill=f)
        text(s, x + 0.12, 1.62, w - 0.24, 0.4, t, size=17, color=c, bold=True, align=PP_ALIGN.CENTER)
        text(s, x + 0.12, 2.05, w - 0.24, 1.1, d, size=13.5, align=PP_ALIGN.CENTER)
        x += w
        if i < len(blocks) - 1:
            arrow(s, x + 0.06, 2.22, 0.36, 0.32)
            x += 0.48

    rows = [
        ["Configurations", "Models", "Role"],
        ["ReAct, thinking off / on", "1.7B · 4B · 8B (6 configs)", "The agent under study"],
        ["Closed-book (no tools)", "1.7B · 4B · 8B", "Memorization / contamination control"],
        ["Single-shot RAG", "1.7B · 4B · 8B", "Simple baseline (one retrieval, then answer)"],
    ]
    table(s, 0.6, 3.55, 7.3, rows, [2.55, 2.35, 2.4], size=13, row_h=0.5)
    box(s, 8.2, 3.55, 4.53, 2.0, fill=NAVY)
    text(s, 8.35, 3.6, 4.25, 1.9,
         [("12 configs × 400 questions × 3 seeds", {"size": 17, "bold": True, "color": WHITE}),
          ("= 14,400 trajectories", {"size": 24, "bold": True, "color": RGBColor(0x5E, 0xEA, 0xD4)}),
          ("Kaggle free tier · T4 × 2 · vLLM fp16 · ≈ 40 GPU-hours planned", {"size": 13, "color": RGBColor(0xCB, 0xD5, 0xE1)})],
         anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    text(s, 0.6, 5.75, 12.13, 1.2,
         ["**Why these choices:** a frozen corpus makes runs reproducible and every gold paragraph findable · required citations make "
          "grounding measurable without an LLM · the closed-book baseline separates retrieval from memorization · 3 sampled seeds "
          "because Qwen3 should not be decoded greedily · the 4B model is the anchor that every analysis finishes first."],
         size=13.5, color=NAVY)
    notes(s, """
[~60 s]
The system is deliberately controlled. The agent searches a frozen corpus of 66 thousand HotpotQA paragraphs, so the gold evidence is always there and runs are reproducible.
It has three tools: BM25 search, read a paragraph, and finish, which must include citations. Requiring citations is what lets me measure grounding without an LLM.
The agent is a hand-written ReAct loop running Qwen3 at 1.7, 4 and 8 billion parameters, with thinking mode on or off, and every step is logged.
Besides the six ReAct configs, each model also runs closed-book, which is my contamination control, and single-shot RAG, which is the simple baseline that Kapoor et al. insist on.
In total: 12 configs, 400 questions, 3 seeds, 14,400 trajectories, on Kaggle's free T4 GPUs, around 40 GPU hours.
""")


def s_eval(prs, n):
    s = base(prs, "Approach", "Every trajectory is scored three ways", n)
    cols = [
        ("A · Outcome-only", AMBER, AMBER_LIGHT,
         ["Exact match and token F1 vs. gold",
          "Reference-guided LLM judge (Qwen3-14B-AWQ): correct / incorrect",
          "pass^k across the 3 seeds"]),
        ("B · Trajectory-aware", BLUE, BLUE_LIGHT,
         ["**Rule-based:** gold-evidence recall, citation precision/recall, grounded, redundant calls, parse failures, recovery after tool errors, token cost",
          "**LLM-judged:** step labels +1 / 0 / −1, first error step, holistic 1–5"]),
        ("C · Human", TEAL, TEAL_LIGHT,
         ["150 stratified trajectories, script-blinded (no model, config, seed or scores)",
          "Same rubric as the LLM judge",
          "Reliability: 30-item re-test after 2 weeks, 40-item external rater → Cohen's κ"]),
    ]
    x = 0.6
    for t, c, f, body in cols:
        card(s, x, 1.5, 3.93, 3.85, t, body, accent=c, fill=f, title_size=18, body_size=14.5)
        x += 4.1
    box(s, 0.6, 5.6, 12.13, 1.15, fill=GREY_LIGHT)
    text(s, 0.85, 5.62, 11.7, 1.1,
         ["**Judges are measured too** (200-trajectory subsample): swap the order of two trajectories (position bias) · "
          "pad thoughts with neutral filler (verbosity bias) · re-judge with Llama-3.1-8B, a different model family (self-preference). "
          "The human labels are a **third measurement with a reliability estimate**, not ground truth by fiat."],
         size=14, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, """
[~60 s]
The core of the method: every trajectory gets three scores.
Family A is outcome-only: exact match, F1, and an LLM judge against the gold answer, since small models often say the right thing in the wrong words. Across seeds it gives pass^k.
Family B looks at the process. Most of it is rule-based and essentially free: did the agent read the gold paragraphs, are its citations correct, did it repeat searches or recover from errors. An LLM judge also labels each step and gives a 1-to-5 process score.
Family C is me, labelling 150 trajectories with the same rubric, blinded by a script. Because I built the agent, I measure my own reliability with a re-test and an external rater.
And I test the LLM judge for the three known biases on a 200-trajectory subsample.
""")


def s_analysis(prs, n):
    s = base(prs, "Approach", "How each research question gets answered", n)
    # RQ1 mini 2x2
    card(s, 0.6, 1.5, 5.95, 2.55, "RQ1 · Disagreement", [], accent=AMBER, fill=AMBER_LIGHT, title_size=16)
    cells = [("RA-GP", "right answer\ngood process", GREEN_LIGHT, GREEN), ("RA-BP", "right answer\nbad process", AMBER_LIGHT, AMBER),
             ("WA-GP", "wrong answer\ngood process", BLUE_LIGHT, BLUE), ("WA-BP", "wrong answer\nbad process", RED_LIGHT, RED)]
    for i, (lab, d, f, c) in enumerate(cells):
        cx, cy = 0.85 + (i % 2) * 1.45, 2.05 + (i // 2) * 0.95
        box(s, cx, cy, 1.38, 0.88, fill=WHITE, line=c)
        text(s, cx, cy + 0.03, 1.38, 0.35, lab, size=13, color=c, bold=True, align=PP_ALIGN.CENTER)
        text(s, cx, cy + 0.35, 1.38, 0.5, d, size=10, color=NAVY, align=PP_ALIGN.CENTER)
    text(s, 3.85, 2.05, 2.6, 1.9,
         ["Off-diagonal rates per config with bootstrap CIs", "RA-BP vs. closed-book accuracy tests the memorization mechanism"],
         size=13, bullet=True)
    card(s, 6.78, 1.5, 5.95, 2.55, "RQ2 · Rankings",
         ["Rank the 6 ReAct configs by accuracy, rule-based process composite, human score, and accuracy within a matched token budget",
          "Kendall τ between rankings (bootstrap CIs) + accuracy-vs-cost Pareto frontier"],
         accent=BLUE, fill=BLUE_LIGHT, body_size=13.5)
    card(s, 0.6, 4.25, 5.95, 2.5, "RQ3 · Validity vs. cost",
         ["Spearman ρ of every automatic metric with the human process score",
          "Plotted against its cost per trajectory (CPU ms vs. judge tokens)",
          "Step-level κ: LLM judge vs. human"],
         accent=TEAL, fill=TEAL_LIGHT, body_size=13.5)
    card(s, 6.78, 4.25, 5.95, 2.5, "RQ4 · Reliability",
         ["pass^k for k = 1…3 (…5 on 4B if time allows)",
          "Diff the seeds that disagree: parse failure, early finish, different query, or reasoning on the same evidence",
          "Paired tests on identical questions (McNemar)"],
         accent=RED, fill=RED_LIGHT, body_size=13.5)
    notes(s, """
[~45 s]
Briefly, how each question is answered. For RQ1, every trajectory lands in one of four cells: right or wrong answer, good or bad process. I report the off-diagonal rates per configuration with confidence intervals, and compare right-answer-bad-process cases against closed-book accuracy to test memorization.
For RQ2, I rank the six ReAct configurations four ways, including accuracy within a fixed token budget, and compute Kendall tau between rankings.
For RQ3, I correlate each automatic metric with the human score and plot that against what the metric costs to compute.
For RQ4, I compute pass^k, and because I have the trajectories, I can diff seeds that disagree and say why they diverged.
All comparisons are paired on the same 400 questions, and confidence intervals resample questions. 400 questions is enough to detect about 10-point differences.
""")


def s_progress(prs, n):
    s = base(prs, "Current progress", "Where the project stands: weeks 1–2 are done", n)
    # 14-week strip
    x0, wcell = 0.6, 12.13 / 14
    for w in range(1, 15):
        done, nxt = w <= 2, w == 3
        f = GREEN if done else (AMBER_LIGHT if nxt else GREY_LIGHT)
        c = WHITE if done else (AMBER if nxt else GREY)
        box(s, x0 + (w - 1) * wcell + 0.03, 1.5, wcell - 0.06, 0.5, fill=f, shape=MSO_SHAPE.RECTANGLE)
        text(s, x0 + (w - 1) * wcell, 1.5, wcell, 0.5, f"W{w}", size=12, color=c, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    for w, lab in ((3, "pre-registration"), (9, "gate"), (14, "final talk")):
        text(s, x0 + (w - 1) * wcell - 0.5, 2.02, wcell + 1.0, 0.3, lab, size=11, color=GREY, align=PP_ALIGN.CENTER)

    card(s, 0.6, 2.5, 3.9, 3.65, "Week 1  ✓",
         ["Read the six core papers",
          "Chose **Qwen3-2504** hybrid checkpoints: thinking on/off within one model",
          "Built the frozen corpus, BM25 index and seeded 400 + 50 question samples; uploaded as a Kaggle Dataset"],
         accent=GREEN, fill=GREEN_LIGHT, body_size=14)
    card(s, 4.72, 2.5, 3.9, 3.65, "Week 2  ✓",
         ["ReAct loop, tools and JSONL step logging",
          "Measured generation throughput for 1.7B / 4B / 8B on a Kaggle T4",
          "First end-to-end pilot trajectories on the GPU",
          "All code and notebooks versioned in git"],
         accent=GREEN, fill=GREEN_LIGHT, body_size=14)
    card(s, 8.83, 2.5, 3.9, 3.65, "Ahead of schedule",
         ["Code for **every later stage** is written: judges, scoring, blinded annotation, RQ1–RQ5 analysis, report",
          "14 automated tests + CPU smoke run of the whole pipeline",
          "Analysis already dry-run on synthetic data (a week-3 deliverable)",
          "Kaggle notebooks for every stage"],
         accent=TEAL, fill=TEAL_LIGHT, body_size=14)
    text(s, 0.6, 6.3, 12.13, 0.55,
         "Remaining work is mostly GPU time and annotation hours, not engineering, which is the right shape for a one-person project.",
         size=14, color=GREY, italic=True)
    notes(s, """
[~50 s]
Here's where things stand on the 14-week schedule. Weeks one and two are done.
In week 1, I read the six core papers, chose the Qwen3 hybrid checkpoints so thinking on versus off is a switch inside one model, and built the frozen corpus, index and seeded question samples.
In week 2, I built the agent loop, tools and logging, measured throughput on Kaggle's T4, and ran the first pilot trajectories end to end.
I'm also ahead on code: judges, scoring, blinded annotation and the full analysis are written and tested, and the analysis already ran on synthetic data, a week-3 task.
So what remains is mostly GPU time and annotation hours rather than engineering.
""")


def s_finding1(prs, n):
    s = base(prs, "Current findings", "Finding 1: the second hop really needs a search", n)
    s.shapes.add_picture(str(HERE / "assets" / "retrieval_headroom.png"), Inches(0.6), Inches(1.45), height=Inches(4.2))
    rows = [
        ["Frozen benchmark", ""],
        ["Corpus", "66,581 paragraphs"],
        ["Eval sample", "400 questions · 320 bridge / 80 comparison"],
        ["Pilot sample", "50 questions, disjoint from eval"],
        ["Excluded", "63 questions with conflicting gold text"],
        ["Answer in gold text", "97.5% of eval questions"],
    ]
    table(s, 7.35, 1.5, 5.38, rows, [2.05, 3.33], size=13, row_h=0.45)
    box(s, 0.6, 5.85, 12.13, 0.95, fill=TEAL_LIGHT)
    text(s, 0.85, 5.88, 11.7, 0.9,
         "One retrieval with the raw question finds **a** gold paragraph for **93%** of questions but **both** for only **51%** "
         "(48% of bridge questions). Most questions need a second-hop search, so the process genuinely varies, and "
         "single-shot RAG is a strong, honest baseline.",
         size=15, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, """
[~70 s]
First finding, from the week-1 data work. The frozen benchmark has 66 thousand paragraphs and 400 hard evaluation questions, plus a separate 50-question pilot set so debugging can't overfit the evaluation. I excluded 63 questions with conflicting gold text, and for 97.5% of questions the answer literally appears in a gold paragraph.
The chart is the important part. Searching with the question itself, BM25 puts at least one gold paragraph in the top 5 for 93% of questions. But it finds both only half the time, and 48% for bridge questions.
That's good for the study: in half the questions the agent needs a second, better-targeted search, so process quality will genuinely differ. And single-shot RAG is a strong baseline the agent has to beat, which keeps the comparison honest.
""")


def s_finding2(prs, n):
    s = base(prs, "Current findings", "Finding 2: the pipeline works end to end on the target setup", n)
    card(s, 0.6, 1.5, 3.85, 4.2, "Verified",
         ["Qwen3's real chat template renders tool calls, tool responses and both thinking modes as the agent expects",
          "Thinking budget: stop at 1,024 tokens, append Qwen's early-exit phrase, then answer",
          "Data → agent → judges → scoring → annotation → report runs end to end"],
         accent=GREEN, fill=GREEN_LIGHT, body_size=14)
    text(s, 4.72, 1.5, 4.1, 0.4, "Week-2 measurements", size=17, color=NAVY, bold=True)
    rows = [
        ["", "Plan's guess", "Measured"],
        ["Qwen3-1.7B tok/s", "–", fill_in("tok_s_1.7b")],
        ["Qwen3-4B tok/s", "300–500", fill_in("tok_s_4b")],
        ["Qwen3-8B tok/s (TP=2)", "–", fill_in("tok_s_8b")],
        ["GPU-h, full grid", "≈ 40", fill_in("gpu_hours_projection")],
        ["Pilot: parse failures", "–", fill_in("pilot_parse_fail")],
        ["Pilot: ended by finish", "–", fill_in("pilot_finish_rate")],
        ["Pilot: budget hit", "–", fill_in("pilot_budget_hit")],
    ]
    table(s, 4.72, 1.95, 4.1, rows, [1.9, 1.1, 1.1], size=12.5, row_h=0.46)
    card(s, 9.05, 1.5, 3.68, 4.2, "Design decisions locked",
         ["BM25 over HotpotQA, not Search-R1's dense retriever",
          "\"Correct\" = LLM judge; EM and F1 always reported",
          "Rankings over the 6 ReAct configs (the only ones with human scores)",
          "Second-family judge: Llama-3.1-8B",
          "RQ5 built, run only if the gate passes"],
         accent=TEAL, fill=TEAL_LIGHT, body_size=13.5)
    box(s, 0.6, 5.95, 12.13, 0.85, fill=GREY_LIGHT)
    text(s, 0.85, 5.97, 11.7, 0.8,
         "Measured throughput replaces the plan's guessed numbers in the compute budget and the pre-registration. "
         "The pilot fixes bugs only and is never used to choose configurations.",
         size=14, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, """
[~60 s]
Second finding: the pipeline works end to end on the target setup.
On the left: prompts render correctly with Qwen3's real chat template, the 1,024-token thinking budget works with Qwen's early-exit phrase, and the pipeline runs from data to report.
In the middle are my week-2 measurements. [Read the numbers: throughput per model, projected GPU hours versus the plan's ~40, and from the pilot: parse failures, runs that ended with finish, and thinking-budget hits.]
On the right are the decisions I locked where the plan and proposal differed: BM25, an LLM judge as primary correctness with exact match always reported, rankings over the six ReAct configs, Llama 3.1 as the second judge, and robustness only after the week-9 gate.
""")


def s_next(prs, n):
    s = base(prs, "Next steps", "Plan for weeks 3–14", n)
    rows = [
        ["Week", "Milestone", "Deliverable"],
        ["3", "**Pre-registration:** freeze questions, hypotheses, metrics, decision rules", "Pre-registration doc + git tag"],
        ["4", "Pilot: 50 pilot questions × 12 configs × 1 seed (bug-fixing only)", "Pilot report"],
        ["5–6", "Main runs in the background, 4B first, then 1.7B and 8B; calibrate and freeze the rubric", "**MVP tables:** RQ1 + RQ4 on 4B"],
        ["7–8", "Baselines; LLM judges and bias tests; human labels for 150 trajectories; recruit external rater", "All first-pass human labels"],
        ["9", "**Gate:** RQ1–RQ4 complete? If yes, run RQ5 + pass^5 on 4B; start the re-test", "Gate decision in repo"],
        ["10", "Full analysis: rankings, validity vs. cost, judge bias, κ / ρ", "Figures v1"],
        ["11", "Full draft; one round of instructor or peer feedback", "Draft + written review"],
        ["12–14", "Revise; release code, trajectories, labels and pre-registration diff; final talk", "8-page paper + artifact"],
    ]
    table(s, 0.6, 1.45, 12.13, rows, [1.0, 7.33, 3.8], size=13, row_h=0.56)
    text(s, 0.6, 6.55, 12.13, 0.4,
         "GPU runs are always queued before the next coding or annotation task, so compute and human work overlap.",
         size=13, color=GREY, italic=True)
    notes(s, """
[~60 s]
Next steps. Week 3 is the most important one: pre-registration. I freeze the questions, hypotheses, metrics and decision rules in git before any main run.
Week 4 is a pilot on the separate 50 questions, for fixing bugs only.
Weeks 5 and 6 are the main runs, 4B first, then 1.7B and 8B, in the background while I freeze the annotation rubric. By week 6 I should have the minimum viable result: RQ1 and RQ4 on 4B.
Weeks 7 and 8 are judges and my 150 human labels. Week 9 is a hard gate: only if RQ1 to RQ4 are complete do I spend time on the robustness experiment and extra seeds.
Then analysis, a full draft with feedback in week 11, and release of the code, trajectories and labels at the end.
""")


def s_risks(prs, n):
    s = base(prs, "Next steps", "Risks, budget and the minimum viable paper", n)
    rows = [
        ["Risk", "Mitigation"],
        ["vLLM support for the T4 GPU", "Pinned version verified in week 2; Hugging Face transformers fallback"],
        ["Small models emit malformed tool calls", "Parse failures tracked separately from reasoning failures; 4B is the anchor"],
        ["HotpotQA is in pretraining data", "Closed-book baseline; memorization test in H1a-mech"],
        ["One annotator grading their own agent", "Script blinding, frozen rubric, re-test, external rater κ"],
        ["Judge and agents are the same family", "Second-family judge on a 200-trajectory subsample"],
        ["A 12-hour Kaggle session is killed", "Checkpoint every 50 questions; resume skips finished work"],
    ]
    table(s, 0.6, 1.45, 7.75, rows, [3.2, 4.55], size=13, row_h=0.6)
    card(s, 8.6, 1.45, 4.13, 2.2, "Compute budget",
         ["≈ 40 GPU-h core (≈ 50 with optionals)",
          "≈ 120 GPU-h of quota in the weeks 5–8 run window",
          "Quota is not the constraint; my attention is"],
         accent=BLUE, fill=BLUE_LIGHT, body_size=13.5)
    card(s, 8.6, 3.85, 4.13, 2.4, "Minimum viable paper",
         ["RQ1 + RQ4 on the 4B model with outcome and rule-based trajectory metrics",
          "Fallback order: drop 8B thinking → drop 1.7B thinking → 300 questions"],
         accent=GREEN, fill=GREEN_LIGHT, body_size=13.5)
    notes(s, """
[~45 s]
Finally, the risks. The biggest technical risk is running vLLM on Kaggle's older T4 GPU; I pinned a version and there's a slower fallback. Small models will produce broken tool calls, so I track those separately rather than letting them dominate every metric. Contamination is handled by the closed-book baseline. The weakest link is that I'm the only annotator and I built the agent, which is why blinding, the re-test and the external rater are built in.
On compute: about 40 GPU hours against roughly 120 available during the run window, so quota isn't the bottleneck; my own time is.
And if the semester goes badly, the minimum viable paper is RQ1 and RQ4 on the 4B model, which is already a complete, defensible result.
""")


def questions_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box(s, 0, 0, 13.333, 7.5, fill=NAVY, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.8, 2.5, 11.7, 1.2, "Questions?", size=54, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    text(s, 0.8, 3.8, 11.7, 1.2,
         ["Main question: on the same trajectories, does the final answer measure the agent?",
          "Code: github.com/lywusa/agent-eval"],
         size=18, color=RGBColor(0xCB, 0xD5, 0xE1), align=PP_ALIGN.CENTER)
    notes(s, """
Thanks. I'm happy to take questions. Useful backup answers:
- Why HotpotQA and not live web search? Reproducibility, and gold evidence labels make grounding measurable without an LLM. Realism is the cost, and it's stated as a limitation.
- Why not trust the LLM judge? Its biases are exactly what H3b measures, and it is compared against human labels with measured reliability.
- Why only 3 seeds? Quota; pass^3 is a weaker test than pass^5, so I add 2 seeds for the 4B anchor if the week-9 gate passes.
""")


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Emu(Inches(13.333))
    prs.slide_height = Emu(Inches(7.5))
    title_slide(prs)
    for i, fn in enumerate([s_background, s_prior, s_rq, s_hyp, s_system, s_eval, s_analysis,
                            s_progress, s_finding1, s_finding2, s_next, s_risks], start=1):
        fn(prs, i)
    questions_slide(prs)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
