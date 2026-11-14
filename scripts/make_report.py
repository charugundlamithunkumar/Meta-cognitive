"""Phase 6: build final-report.pdf from the committed results (no GPU needed).

usage: python3 scripts/make_report.py
Reads results/main_results.json, results/figure_data.json, results/secondary.json and results/figures/*.png.
"""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Image,
                                PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
M = json.loads((RES / "main_results.json").read_text())
F = json.loads((RES / "figure_data.json").read_text())
SEC = json.loads((RES / "secondary.json").read_text())

FD = "/usr/share/fonts/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", FD + "DejaVuSans-BoldOblique.ttf"))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B"); addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")

INK, MUTED, ACCENT = colors.HexColor("#1f2933"), colors.HexColor("#52606d"), colors.HexColor("#1d4e89")
RULE, HEADBG, BOXBG = colors.HexColor("#cbd2d9"), colors.HexColor("#e4ebf3"), colors.HexColor("#f4f7fa")
GOOD, PART, BAD, NEU = (colors.HexColor(c) for c in ("#e3f1e6", "#fbf1d9", "#f8e1df", "#eceff2"))

body = ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13, textColor=INK)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.6)
cap = ParagraphStyle("cap", parent=body, fontSize=7.8, leading=10.2, textColor=MUTED, spaceAfter=8)
cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10.4)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)
title = ParagraphStyle("title", fontName="DV-B", fontSize=17, leading=21, textColor=INK, spaceAfter=4)
subtitle = ParagraphStyle("subtitle", fontName="DV", fontSize=10, leading=14, textColor=MUTED, spaceAfter=10)
h1 = ParagraphStyle("h1", fontName="DV-B", fontSize=12.5, leading=16, textColor=ACCENT, spaceBefore=12, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="DV-B", fontSize=10, leading=13.5, textColor=INK, spaceBefore=8, spaceAfter=3)

W = A4[0] - 36 * mm


def P(t, s=body):
    return Paragraph(t, s)


def bl(items, s=bullet):
    return [Paragraph(i, s, bulletText="•") for i in items]


def table(rows, widths, header=True, shade=None):
    data = [[Paragraph(c, cellb if header and r_i == 0 else cell) for c in r] for r_i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEADBG))
    for (r, c), col in (shade or {}).items():
        st.append(("BACKGROUND", (c, r), (c, r), col))
    t.setStyle(TableStyle(st))
    return t


def box(flow, bg=BOXBG, bar=ACCENT):
    t = Table([[flow]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def fig(name, caption, width=W):
    p = RES / "figures" / name
    iw, ih = ImageReader(str(p)).getSize()
    return KeepTogether([Image(str(p), width=width, height=width * ih / iw), Spacer(1, 2), P(caption, cap)])


def ci(d, k1="diff", lo="lo", hi="hi", n=3):
    return f"{d[k1]:+.{n}f} [{d[lo]:+.{n}f}, {d[hi]:+.{n}f}]"


def footer(canv, doc):
    canv.saveState()
    canv.setFont("DV", 7.5)
    canv.setFillColor(MUTED)
    canv.drawString(18 * mm, 10 * mm, "Results report: Catching Silent Collapse (IBM Granite 4.2 8B)")
    canv.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canv.restoreState()


HA, HB, HC, HD, HE = (M[h] for h in ("HA", "HB", "HC", "HD", "HE"))
ACC = F["accuracy_per_arm"]
S = []

# ------------------------------------------------------------------ title + summary
S.append(P("Catching Silent Collapse", title))
S.append(P("Results report for instructor review · IBM Granite 4.2 8B on Long-Horizon-Execution · "
           "pre-registered main runs, 13–14 Nov 2026", subtitle))
S.append(box([
    P("<b>Summary.</b> We tested whether signals that already work, token log-probabilities and self-audits run in "
      "a fresh context, can be used <i>inside the agent loop</i> to stop the silent collapse caused by "
      "inherited errors. The test used 9 arms × 32 held-out trajectories × 300 turns, analysed with pre-registered rules. "
      f"Baseline accuracy is {ACC['A0']['point']:.3f}: the first error comes at turn "
      f"{SEC['turns_to_first_error']['A0']:.1f} and is then carried forward. "
      f"A <b>logit-gated re-compute</b> beats an identical random gate (+{HA['acc_A1b_minus_A2']['diff']:.3f}) and "
      f"delays the first error to turn {SEC['turns_to_first_error']['A1b']:.0f}. "
      f"A <b>fresh-context thinking audit</b> gives the largest gain (accuracy {ACC['A3']['point']:.3f}). "
      "Granite's audit asked <i>inside</i> the history did not get worse late in the run, against our prediction. "
      "The forecast's failure to notice decline is <b>input-starved</b>: given a checker's count, forecast error falls from "
      f"{HC['mae']['none']:.2f} to {HC['mae']['true']:.2f}. "
      "Removing the history but keeping the agent's own total leaves accuracy unchanged, so the collapse is inheritance. "
      "Even the true-total control still drifts downward, which we trace (exploratory) to the running total "
      "growing over the run. The bottleneck is not detection: detectors catch 87–100% of errors. It is the agent's "
      "new-slip rate of about 44% per turn."),
]))
S.append(Spacer(1, 8))
verdict_rows = [
    ["", "Hypothesis", "Decisive result (95% CI, paired bootstrap)", "Holm p", "Verdict"],
    ["HA", "Catch the seed: a logit-gated re-compute beats a random gate",
     f"A1b − A2 accuracy {ci(HA['acc_A1b_minus_A2'])}; A1b − A0 {ci(HA['acc_A1b_minus_A0'])}; "
     f"error chains −{-HA['chains_A1b_minus_A2']['diff']:.1f} vs A2", "&lt; 0.001", "<b>Supported</b>"],
    ["HB", "Audit outside the history: a fresh audit stays accurate, an in-history audit degrades",
     f"A3 − A0 accuracy {ci(HB['acc_A3_minus_A0'])}; ΔJ A3 {ci(HB['delta_A3'], 'point')}; "
     f"ΔJ A4 {ci(HB['delta_A4'], 'point')} (not entirely &lt; 0)", f"{HB['p_holm']:.3f}",
     "<b>Partial</b>: A3 works, in-history decline not shown"],
    ["HC", "Awareness is input-starved",
     f"MAE true − none {ci(HC['true_minus_none'])}; true − placebo {ci(HC['true_minus_placebo'])}; "
     f"Spearman {HC['spearman_true']['point']:.2f} [{HC['spearman_true']['lo']:.2f}, {HC['spearman_true']['hi']:.2f}]",
     f"{HC['p_holm']:.4f}", "<b>Supported</b>"],
    ["HD", "Collapse is inheritance, not context length",
     f"D1 − A0 accuracy {HD['acc_D1_minus_A0']['diff']:+.3f}, 90% CI [{HD['acc_D1_minus_A0']['lo90']:+.3f}, "
     f"{HD['acc_D1_minus_A0']['hi90']:+.3f}] ✓; D2 slope {HD['D2_slope']['mean']*1e4:+.2f}×10⁻⁴/turn "
     f"[{HD['D2_slope']['lo']*1e4:+.2f}, {HD['D2_slope']['hi']*1e4:+.2f}], band ±5×10⁻⁴ ✗", "—",
     "<b>Not supported</b> (the D1 part holds)"],
    ["HE", "The logit format effect is a scoring artifact",
     f"AUROC inline − bare = {HE['diff_inline_pre_minus_A0']['point']:+.3f} "
     f"[{HE['diff_inline_pre_minus_A0']['lo']:+.3f}, {HE['diff_inline_pre_minus_A0']['hi']:+.3f}]: condition not met", "—",
     "<b>Not reproduced</b> (excluded from Holm, as pre-registered)"],
]
S.append(table(verdict_rows, [W * .06, W * .25, W * .41, W * .09, W * .19],
               shade={(1, 4): GOOD, (2, 4): PART, (3, 4): GOOD, (4, 4): BAD, (5, 4): NEU}))
S.append(P("Holm correction over the four tested hypotheses (HA–HD). HD's p is the TOST p of its D1 part; its verdict "
           "uses both parts (see §6).", cap))

# ------------------------------------------------------------------ 1. design
S.append(P("1. Question and design", h1))
S.append(P("<b>Problem.</b> Long-horizon agents fail silently. One early slip is carried forward (an <i>inherited</i> "
           "error), and the agent's self-reports do not notice. The instructor's analysis showed that token "
           "log-probabilities detect the original slips, and that audits work in short contexts. We test whether these "
           "signals can be used in the loop to stop the collapse, and what that says about its mechanism."))
S.append(P("<b>Task and data.</b> Long-Horizon-Execution (Sinha et al., ICLR 2026; Apache-2.0; pinned revision "
           "<font face='DV' size='8'>7e4901f</font>). The agent keeps a running sum over a 100-key dictionary, one key "
           "per turn (K = 1), for T = 300 turns. The true total is known at every turn. Splits: calibration 0–19 "
           "(pilot only), <b>test 20–51</b>, reserve 52–99 (untouched)."))
S.append(P("<b>Model.</b> IBM Granite 4.2 8B (revision <font size='8'>a704214</font>), bf16, vLLM 0.24.0, on one "
           "RTX 6000 Ada. The agent decodes greedily without thinking, with replies forced to an integer by a regex; "
           "raw top-20 logprobs are logged. Re-computes use low-effort thinking. Audits use full thinking (budget "
           "8,192), with the model card's sampling and fixed seeds."))
S.append(table([
    ["Arm", "What changes"],
    ["A0", "Baseline; side questions probe P(Yes), verbal confidence and 10-turn forecasts (no / true / placebo evidence)"],
    ["A0_inline", "Reply is <i>total | confidence</i> (HE)"],
    ["A1a / A1b / A2", "Re-compute gate fired by: format only / format + logit threshold τ = −0.138 / at random, at A1b's realized rate (0.450)"],
    ["A3 / A4", "Thinking audit every 5 turns: in a fresh context / inside the full history; a WRONG verdict adds a correction note"],
    ["D1 / D2", "No history; a state line gives the agent's own last total / the true total (oracle)"],
], [W * .17, W * .83]))
S.append(Spacer(1, 4))
S.append(P("<b>Statistics (pre-registered in <font size='8'>prereg.md</font>, commit 6c6ac8c, before any test run).</b> "
           "The trajectory is the unit. Paired bootstrap over the 32 trajectories, with 10,000 resamples and 95% "
           "percentile CIs. Holm correction across the primary tests. AUROC and auditor J use trajectory-bootstrap "
           "CIs. The run script refused to start unless every config and source file matched its registered "
           "SHA-256. Each arm ran in one pass, with no resume and no stitching."))

# ------------------------------------------------------------------ 2. results overview
S.append(P("2. Results overview", h1))
S.append(fig("fig2_accuracy_per_arm.png", "Figure 1. Accuracy over 300 turns per arm (mean of 32 trajectories, 95% CI). "
             "D2 is an oracle control: it is given the true total."))
rows = [["Arm", "Accuracy [95% CI]", "First error (turn)", "New-slip rate by turn bin (0–49 / 50–99 / 100–199 / 200–299)", "Tokens / turn"]]
for a in ["A0", "A0_inline", "A1a", "A1b", "A2", "D1", "D2", "A3", "A4"]:
    v = ACC[a]
    sr = " / ".join(f"{x:.2f}" for x in SEC["slip_rate_by_bin"][a])
    rows.append([a, f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]", f"{SEC['turns_to_first_error'][a]:.1f}", sr,
                 f"{SEC['mean_generated_tokens_per_turn'][a]:.0f}"])
S.append(table(rows, [W * .12, W * .22, W * .14, W * .38, W * .14]))
S.append(P("Accuracy CIs: 2,000 resamples (figures); verdict CIs use 10,000. The new-slip rate is the share of turns "
           "whose committed total is ≠ prev + v; in A1b/A2 it is measured after the gate. Tokens are generated "
           "tokens per turn, including re-computes and audits.", cap))
S.append(fig("fig1_accuracy_by_turn.png", "Figure 2. Accuracy by turn (10-turn rolling mean over 32 trajectories). "
             "(a) In-loop interventions. (b) Compaction controls."))
S.append(fig("fig3b_error_types_by_turn.png", "Figure 3. Share of correct, inherited and newly slipped turns by "
             "turn (10-turn windows). Without an intervention, the correct share hits zero within about 30 turns; "
             "afterwards every turn is either a new slip or an inherited error."))

# ------------------------------------------------------------------ 3. per hypothesis
S.append(P("3. Results by hypothesis", h1))
g = SEC["gate"]
S.append(P("HA: catch the seed (supported)", h2))
S += bl([
    f"The A1b gate fired on {g['A1b']['fire_rate']:.1%} of turns. It caught <b>{g['A1b']['recall_on_local_slips']:.1%}</b> "
    f"of locally wrong replies, and it also fired on {g['A1b']['false_trigger_rate']:.1%} of locally correct ones. The random "
    f"gate A2 fired at the same rate ({g['A2']['fire_rate']:.1%}) but caught only {g['A2']['recall_on_local_slips']:.1%}.",
    f"The re-compute (low-effort thinking, about 20 tokens) was locally correct in {g['A1b']['recompute_locally_correct']:.1%} "
    "of fired turns. So the gate's advantage comes entirely from <i>where</i> it fires.",
    f"Live-gate AUROC is {HA['gate_auroc']['point']:.3f} [{HA['gate_auroc']['lo']:.3f}, {HA['gate_auroc']['hi']:.3f}]. "
    f"Error chains per trajectory fall by {-HA['chains_A1b_minus_A2']['diff']:.1f} against A2 and by "
    f"{-HA['chains_A1b_minus_A0']['diff']:.1f} against A0. The first error moves from turn "
    f"{SEC['turns_to_first_error']['A0']:.1f} (A0) to {SEC['turns_to_first_error']['A1b']:.1f}.",
    "<b>Why accuracy is still only 0.067:</b> the gate misses about 13% of slips, so a new error still gets through on "
    "about 5% of turns, and each one is carried forward for the rest of the run. The gate prevents new chains; it cannot repair a chain that has already started.",
])
S.append(P("HB: audit outside the history (partial)", h2))
J3, J4 = F["auditor"]["A3"], F["auditor"]["A4"]
S.append(table([["Turn bin", "0–49", "50–99", "100–199", "200–299"],
                ["J, A3 fresh"] + [f"{x[0]:.3f} [{x[1]:.2f}, {x[2]:.2f}]" for x in J3["J"]],
                ["J, A4 in-history"] + [f"{x[0]:.3f} [{x[1]:.2f}, {x[2]:.2f}]" for x in J4["J"]],
                ["Unfinished, A3"] + [f"{x:.1%}" for x in J3["unfinished"]],
                ["Unfinished, A4"] + [f"{x:.1%}" for x in J4["unfinished"]]],
               [W * .2] + [W * .2] * 4))
S.append(Spacer(1, 4))
S += bl([
    f"The fresh audit raises accuracy by {ci(HB['acc_A3_minus_A0'])}, the largest effect of any arm. Both auditors "
    "flag ≥ 95% of wrong totals at every position.",
    f"<b>Against our prediction, the in-history audit did not degrade.</b> Its ΔJ is {HB['delta_A4']['point']:+.3f} with "
    f"a CI upper bound of exactly {HB['delta_A4']['hi']:.3f}. It does cost more: it fails to finish more often "
    f"({HB['unfinished_A4']:.1%} against {HB['unfinished_A3']:.1%}) and uses more thinking tokens (3,382 against 3,030 per "
    "audit). Exploratory: A3 − A4 accuracy is +0.017 [−0.001, +0.037].",
    "<b>Why accuracy stays near 0.12 with a near-perfect auditor:</b> audits run every 5 turns, and the agent makes a "
    "new slip on about 44% of turns. So a correction rarely survives until the next audit. The audit lowers "
    "inherited errors (0.540 → 0.434) but cannot lower new slips.",
])
S.append(fig("fig4_auditor_by_bin.png", "Figure 4. (a) Auditor J by turn bin with trajectory-bootstrap 95% CIs. "
             "(b) Share of audits that ran out of the 8,192-token thinking budget."))
S.append(P("HC: awareness is input-starved (supported)", h2))
S += bl([
    f"Forecast MAE over {HC['n_checkpoints']} checkpoints in {HC['n_traj']} trajectories: no evidence "
    f"<b>{HC['mae']['none']:.3f}</b>, true checker count <b>{HC['mae']['true']:.3f}</b>, placebo count "
    f"<b>{HC['mae']['placebo']:.3f}</b>. The model reacts to the <i>count</i>, not just to being told a checker exists.",
    "Caveat: forecasts are close to 0 or 100, and the realized next-10 accuracy is 0 at most checkpoints. "
    "The result mainly shows that the model believes evidence of failure when it is given some.",
])
S.append(fig("fig5_hc_forecasts.png", "Figure 5. (a) Forecast error per evidence condition (95% CI). "
             "(b) Each checkpoint: forecast against realized accuracy over the next 10 turns."))
S.append(P("HD: inheritance vs. context length (not supported)", h2))
S += bl([
    f"<b>D1</b> (no history, own total) matches A0 in accuracy: {HD['acc_D1_minus_A0']['diff']:+.3f}, 90% CI inside ±0.05. "
    "Without the long history the agent makes <i>fewer</i> new slips (0.302 against 0.446; "
    "−0.144 [−0.208, −0.087]), but every wrong total is still carried forward, so accuracy does not move. "
    "<b>The collapse is inheritance.</b>",
    f"<b>D2</b> (true total supplied) reaches {HD['acc']['D2']:.3f} but still declines by "
    f"{HD['D2_slope']['mean']*300:.2f} over 300 turns, outside the pre-registered flatness band. So HD's second "
    "part fails.",
    "<b>Exploratory (post hoc):</b> the mean |total| grows from about 230 to about 985 across the run. With "
    "|total| below 1000, D2 accuracy is 0.69 / 0.67 / 0.66 / 0.64 by bin; with 4-digit totals it is about 0.37–0.40 "
    "in every bin with at least 20 such turns. The drift tracks number size more than position (Figure 6).",
])
S.append(fig("fig7_d2_by_magnitude.png", "Figure 6 (exploratory). D2 accuracy by turn bin, split by the size of "
             "the true running total; n = turns.", width=W * .78))
S.append(P("HE: logit format effect (not reproduced)", h2))
S += bl([
    f"AUROC of the minimum answer-token logprob: A0 bare {HE['auroc_A0']['point']:.3f} "
    f"[{HE['auroc_A0']['lo']:.3f}, {HE['auroc_A0']['hi']:.3f}]; A0_inline {HE['auroc_inline_pre']:.3f}. The inline "
    "format did not lower the signal, so the condition HE depends on did not occur.",
    "Under forced-integer decoding, the tokens before '|' and the final-integer tokens are the same tokens, so the "
    "two scoring rules give the same AUROC.",
])
S.append(fig("fig6_logit_auroc.png", "Figure 7. Type-2 AUROC (locally correct vs. new slip) of the minimum "
             "answer-token logprob; dashed line = chance.", width=W * .85))

# ------------------------------------------------------------------ 4. synthesis
S.append(P("4. What this shows", h1))
S += bl([
    "<b>The collapse is inheritance.</b> Removing 300 turns of history does not change accuracy as long as the "
    "agent's own wrong total is carried forward (D1). Giving the true total lifts accuracy from 0.014 to 0.604 (D2).",
    f"<b>The model's self-reports are blind, but its logits are not.</b> On A0: P(Yes) AUROC "
    f"{SEC['probe_auroc']['ptrue']:.2f} (chance), verbal confidence {SEC['probe_auroc']['verbal']:.2f}; inline "
    "confidence was 100 on every one of 9,600 turns. Token logprobs reach 0.79–0.90, and they work when used "
    "in the loop (HA).",
    "<b>Detection is not the bottleneck; the new-slip rate is.</b> The gate catches 87% of slips, and the "
    "audits catch 95–100% of wrong totals. But the agent slips on about 44% of turns, and every miss is inherited. "
    "Gains are therefore capped by how often the check runs (audits) or by what it misses (gate).",
    "<b>Cost.</b> The gate costs about 4× the baseline's tokens per turn (11.8 against 2.8); the audits cost "
    "about 220–270× (612–762 tokens per turn). Per extra token, the gate is far cheaper; in absolute gain, the "
    "audit is larger.",
])

# ------------------------------------------------------------------ 5. secondary
S.append(P("5. Secondary findings (not used for verdicts)", h1))
S += bl([
    f"<b>Reproducibility.</b> A1a's gate never fires (forced integers are never malformed), so it should replay A0. "
    f"Only {SEC['A0_vs_A1a_identical_reply_share']:.1%} of replies are identical "
    f"({SEC['A0_vs_A1a_identical_first_10_turns']:.1%} in the first 10 turns): vLLM batching nondeterminism "
    "under greedy decoding means the two runs follow different error paths after the first differing reply. Their "
    "accuracies agree (0.014 against 0.018), and A1b − A1a ≈ A1b − A0, so conclusions hold at the arm level.",
    f"<b>The slip rate rises with position in A0</b> (0.25 → 0.54 across bins), but is flat in D1 (about 0.30). "
    "The long history raises the rate of <i>new</i> slips, even though this does not change accuracy.",
    f"<b>Retrieval-like slips</b> (the step added a <i>different</i> dictionary value than the right one): "
    f"{SEC['retrieval_like_share_of_slips']['A0']:.1%} of A0 slips against "
    f"{SEC['retrieval_like_share_of_slips']['D2']:.1%} in D2 (exploratory).",
    "<b>Data quality.</b> 0 non-answers in all 86,400 rows. An all-row consistency check (truth = running sum of "
    "dictionary values; label agrees with the committed total; D2's state line equals the truth) found no "
    "problems. A hand spot-check of 18 random rows also found none.",
])

# ------------------------------------------------------------------ 6. deviations
S.append(P("6. Changes after pre-registration", h1))
S.append(P("All are logged in <font size='8'>implementation-plan.md</font> §13. None changes a verdict."))
S += bl([
    "The analysis script was committed 10 minutes after launch, before any test output existed. It implements "
    "prereg §3–§4 as written.",
    "HD's Holm p is the TOST p of its D1 part. The D2 slope rule is a CI band without a p-value.",
    "The D2 analysis by number size is post hoc and reported as exploratory.",
    "The prereg §5 contrast \"slip rate before vs. after the first error\" turned out degenerate (the before-rate is 0 "
    "by definition); we report across-arm slip rates instead.",
    "Figures use 2,000 bootstrap resamples; the verdicts use the registered 10,000.",
])

# ------------------------------------------------------------------ 7. limitations
S.append(P("7. Limitations", h1))
S += bl([
    "One model and one synthetic task. Forced-integer decoding changes the output channel. The other group "
    "members' runs (Qwen3-8B, Llama) with the same protocol test whether the results generalize.",
    "The baseline sits near the floor (first error at turn about 4), which compresses accuracy differences and "
    "leaves few correct audited totals late in the run. The FPR side of the late-bin J is therefore thin.",
    "On this task the running total grows with position, so the pre-registered position tests (D2 slope, ΔJ) "
    "cannot separate position from number size.",
    "Logit AUROC on the test set (A0 0.79) is lower than in the calibration pilot (0.88–0.91). τ was not re-tuned.",
    "HC rests on 56 checkpoints, with forecasts close to 0 or 100.",
    "GPU 0 was shared, so wall-clock times are not comparable across arms (token counts are).",
])

# ------------------------------------------------------------------ 8. next steps
S.append(P("8. Recommended next steps", h1))
S += bl([
    "<b>Audit frequency.</b> Rerun A3 with m = 1–2 on the reserve set (52–99). If new slips are the bottleneck, "
    "accuracy should rise towards the D2 ceiling, at higher cost.",
    "<b>Combine the gate and the audit (A1b + A3).</b> The gate removes most new chains cheaply; the audit repairs "
    "the ones it misses. These are complementary, so the combination is the natural candidate fix.",
    "<b>Separate position from number size.</b> Use a variant where the total stays bounded (e.g. sums modulo "
    "1000, or keys balanced to keep |total| small), then repeat the D2 slope and ΔJ tests.",
    "<b>Compare across models</b> with the teammates' Qwen3-8B and Llama runs under the identical pre-registered "
    "protocol (same configs, splits and analysis scripts).",
])

# ------------------------------------------------------------------ 9. review + reproducibility
S.append(P("9. Questions for the instructor's review", h1))
S += bl([
    "Is the HB partial verdict, with A4's ΔJ upper bound at exactly 0, best reported as stated, or should "
    "the borderline be stressed further?",
    "Is the size explanation of the D2 drift strong enough to motivate the bounded-total follow-up, or should "
    "HD simply stand as \"not supported\"?",
    "Which follow-up should come first: audit frequency, gate + audit, or the bounded-total task?",
])
S.append(P("10. Reproducibility", h1))
S.append(P("Repository <font size='8'>github.com/charugundlamithunkumar/Meta-cognitive</font>. Main-run stamp "
           "<font size='8'>20261113-173033</font>; code = pre-registration commit 6c6ac8c. The raw outputs of every arm "
           "are in <font size='8'>results/raw_main_20261113-173033/</font> (about 10 MB). Unpack them into "
           "<font size='8'>runs/</font> and run <font size='8'>analyze_main.py</font>, <font size='8'>figures.py</font>, "
           "<font size='8'>secondary.py</font> and <font size='8'>make_report.py</font> to rebuild every number and "
           "figure in this report without a GPU. Full write-up: <font size='8'>results/main_results.md</font>."))

doc = SimpleDocTemplate(str(ROOT / "final-report.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=16 * mm, bottomMargin=18 * mm, title="Catching Silent Collapse: Results Report",
                        author="Granite 4.2 8B group member", subject="Pre-registered results, Granite 4.2 8B")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote final-report.pdf")
