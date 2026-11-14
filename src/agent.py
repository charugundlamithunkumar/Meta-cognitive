"""The turn loop for every arm (implementation-plan.md §6).

Arms: A0, A0_inline, A1a, A1b, A2, A3, A4, D1, D2.
All trajectories advance in lockstep; each turn is one batched vLLM call (+ batched forks).
The chat format is derived from the model's own official chat template (see Chat), so the
same code works for any model with an `enable_thinking` switch (our model: Granite 4.2 8B).
"""
import json
import math
import random
import re
import time
from pathlib import Path

from . import prompts as PR
from .labels import parse_reply, label_turn, retrieval_like, next_prev
from .metrics import token_stats

VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|WRONG)(?:\s*,\s*total\s*=\s*(-?\d+))?", re.I)
INT_RE = re.compile(r"-?\d+")

HISTORY_ARMS = {"A0", "A0_inline", "A1a", "A1b", "A2", "A3", "A4"}
GATE_ARMS = {"A1a", "A1b", "A2"}
AUDIT_ARMS = {"A3", "A4"}
COMPACT_ARMS = {"D1", "D2"}
PROBE_ARMS = {"A0", "A0_inline"}


class Chat:
    """Segment templates derived from the tokenizer's official chat template.

    We render small conversations containing sentinel strings with apply_chat_template and
    cut them into: system segment, user segment, past-assistant segment, and the
    non-thinking / thinking generation prompts. tests/test_core.py checks that composing
    these segments reproduces apply_chat_template exactly (string and token ids).
    """
    S, U, U2, X = "SENTINELSYSTEM", "SENTINELUSERA", "SENTINELUSERB", "SENTINELASSISTANT"

    def __init__(self, tok):
        self.tok = tok
        S, U, U2, X = self.S, self.U, self.U2, self.X

        def r(msgs, gen=False, think=False):
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=think)

        sys_only = r([dict(role="system", content=S)])
        base = r([dict(role="system", content=S), dict(role="user", content=U)])
        assert base.startswith(sys_only), "system segment is not a prefix"
        self.sys_tpl = sys_only
        self.user_tpl = base[len(sys_only):]
        msgs = [dict(role="system", content=S), dict(role="user", content=U)]
        self.gen_nt = r(msgs, gen=True, think=False)[len(base):]
        self.gen_th = r(msgs, gen=True, think=True)[len(base):]
        two = r(msgs + [dict(role="assistant", content=X), dict(role="user", content=U2)])
        user2 = self.user_tpl.replace(U, U2)
        assert two.startswith(base) and two.endswith(user2), "unexpected template structure"
        self.asst_tpl = two[len(base):len(two) - len(user2)]
        for name, seg, s in [("system", self.sys_tpl, S), ("user", self.user_tpl, U), ("assistant", self.asst_tpl, X)]:
            assert seg.count(s) == 1, f"{name} segment must contain its sentinel once"
        assert self.gen_nt != self.gen_th, "template has no enable_thinking switch"

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def system(self, s):
        return self.enc(self.sys_tpl.replace(self.S, s))

    def user(self, u):
        return self.enc(self.user_tpl.replace(self.U, u))

    def asst(self, x):
        return self.enc(self.asst_tpl.replace(self.X, x))

    def gen(self, think=False):
        return self.enc(self.gen_th if think else self.gen_nt)


def answer_token_info(co, tok, eos_ids, inline):
    """Per-token chosen logprobs and top-k lists for the answer, split into
    all (non-EOS), pre ('|' excluded and after), fin (tokens of the reported integer).
    Tokens that start at or after the end of the returned text (the line-break stop token)
    are excluded, so statistics cover only the answer itself."""
    ids = [i for i in co.token_ids if i not in eos_ids]
    strs = [tok.decode([i]) for i in ids]
    keep, pos, n_text = [], 0, len(co.text)
    for q, s in enumerate(strs):
        if pos < n_text or (n_text == 0 and q == 0):
            keep.append(q)
        pos += len(s)
    ids = [ids[q] for q in keep]
    strs = [strs[q] for q in keep]
    chosen = [co.logprobs[q][ids[q]].logprob for q in range(len(ids))]
    tops = [[lp.logprob for lp in co.logprobs[q].values()] for q in range(len(ids))]
    text = "".join(strs)
    offs, pos = [], 0
    for s in strs:
        offs.append((pos, pos + len(s)))
        pos += len(s)
    pipe = text.find("|") if inline else -1
    left_end = pipe if pipe >= 0 else len(text)
    pre = [q for q, (a, b) in enumerate(offs) if a < left_end]
    ints = list(INT_RE.finditer(text[:left_end]))
    if ints:
        s0, s1 = ints[-1].span()
        fin = [q for q, (a, b) in enumerate(offs) if a < s1 and b > s0]
    else:
        fin = []
    pick = lambda qs: ([chosen[q] for q in qs], [tops[q] for q in qs])
    # Renormalized over number-like tokens (digits / minus) among the top-k at each position:
    # isolates numeric uncertainty from mass on words (relevant under constrained decoding).
    renorm = []
    for q in fin:
        num = []
        for tid, lp in co.logprobs[q].items():
            s = tok.decode([tid])
            if s.strip() and re.fullmatch(r"\s?-?\d*", s):
                num.append(lp.logprob)
        if num:
            den = math.log(sum(math.exp(x) for x in num))
            renorm.append(chosen[q] - den)
    return dict(
        all=token_stats(chosen, tops),
        pre=token_stats(*pick(pre)),
        fin=token_stats(*pick(fin)),
        fin_renorm_lp_min=(min(renorm) if renorm else float("nan")),
        tok_strs=strs,
        tok_lps=[round(x, 5) for x in chosen],
    )


def p_from_first_token(co, pos_ids, neg_ids):
    lps = co.logprobs[0]
    py = sum(math.exp(lps[i].logprob) for i in pos_ids if i in lps)
    pn = sum(math.exp(lps[i].logprob) for i in neg_ids if i in lps)
    return py / (py + pn) if py + pn > 0 else float("nan")


def parse_audit(text):
    if "</think>" not in text:
        return dict(finished=False, verdict=None, total=None)
    tail = text.rsplit("</think>", 1)[1]
    ms = list(VERDICT_RE.finditer(tail))
    if not ms:
        return dict(finished=False, verdict=None, total=None)
    m = ms[-1]
    return dict(finished=True, verdict=m.group(1).upper(),
                total=int(m.group(2)) if m.group(2) is not None else None)


def _run_group(cfg, trajs, dictionary, llm, tok, fo, log, t_start, label):
    """Run one group of trajectories in lockstep. Rows are flushed after every turn.
    Returns one summary dict per trajectory."""
    from vllm import SamplingParams

    arm = cfg["arm"]
    K = cfg["K"]
    PR.PROMPT_VERSION["current"] = cfg.get("prompt_version", 1)
    inline = arm == "A0_inline"
    ch = Chat(tok)
    eos_ids = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                           tok.convert_tokens_to_ids("<|endoftext|>")]
               if isinstance(i, int) and i != tok.unk_token_id}
    stop = list(eos_ids)
    dict_vals = set(dictionary.values())
    yes_ids = [ch.enc(w)[0] for w in ["Yes", "yes", " Yes"]]
    no_ids = [ch.enc(w)[0] for w in ["No", "no", " No"]]
    abc_ids = [ch.enc(w)[0] for w in ["A", "B", "C"]]

    # Replies end at the first line break (the answer is the first line); see plan §13.
    # With cfg["constrain"], replies are forced by regex to be an integer (inline: 'N | C'),
    # so format failures cannot occur or enter the history; logprobs stay raw (pre-constraint).
    def so(regex):
        if not cfg.get("constrain"):
            return {}
        from vllm.sampling_params import StructuredOutputsParams
        return dict(structured_outputs=StructuredOutputsParams(regex=regex))
    INT_RX = r"-?[0-9]{1,6}"
    ans_rx = INT_RX + r" \| [0-9]{1,3}" if inline else INT_RX
    sp_ans = SamplingParams(temperature=0, max_tokens=24 if inline else 16, logprobs=20, stop_token_ids=stop,
                            stop=["\n"], **so(ans_rx))
    sp_num = SamplingParams(temperature=0, max_tokens=6, stop_token_ids=stop, stop=["\n"], **so(r"[0-9]{1,3}"))
    sp_lp1 = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    sp_rec = SamplingParams(temperature=0, max_tokens=16, stop_token_ids=stop, stop=["\n"], **so(INT_RX))

    def sp_audit(seed):
        # Thinking audits use the model card's recommended sampling (Granite 4.2: temperature 1.0,
        # top_p 0.95), seeded per (trajectory, turn) for reproducibility.
        a = cfg["audit_sampling"]
        return SamplingParams(temperature=a["temperature"], top_p=a["top_p"], top_k=a.get("top_k", -1),
                              max_tokens=cfg["think_budget"], stop_token_ids=stop, seed=seed)

    sys_agent = ch.system(PR.agent_system(dictionary, K, inline))
    sys_rec = ch.system(PR.recompute_system(dictionary))
    sys_aud = ch.system(PR.audit_system(dictionary))
    checkpoints = set(cfg.get("checkpoints", []))

    B = len(trajs)
    T = len(trajs[0].turns)
    S = [dict(ctx=list(sys_agent), prev=0, V=0, t0=-1, note=None, kinds=[], n_gate=0, n_audit=0, n_correct=0,
              rng=random.Random(cfg.get("seed", 0) * 100003 + tr.sample_id)) for tr in trajs]
    probes_on = arm in PROBE_ARMS and cfg.get("probes", True)

    for t in range(T):
        # ---------- 1. answer
        user_txt, prompts = [], []
        for b, tr in enumerate(trajs):
            turn = tr.turns[t]
            if arm in COMPACT_ARMS:
                state = S[b]["prev"] if arm == "D1" else (trajs[b].turns[t - 1].truth if t > 0 else 0)
                u = PR.compact_turn(state, turn.keys)
                prompts.append(sys_agent + ch.user(u) + ch.gen())
            else:
                u = PR.turn_text(turn.keys, S[b]["note"])
                prompts.append(S[b]["ctx"] + ch.user(u) + ch.gen())
            user_txt.append(u)
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp_ans, use_tqdm=False)

        rows = []
        for b, o in enumerate(outs):
            co = o.outputs[0]
            turn = trajs[b].turns[t]
            raw = co.text.strip()
            val, wf, conf = parse_reply(raw, inline)
            info = answer_token_info(co, tok, eos_ids, inline)
            prev_used = (S[b]["prev"] if arm != "D2" else (trajs[b].turns[t - 1].truth if t > 0 else 0))
            rows.append(dict(arm=arm, sample_id=trajs[b].sample_id, turn=t, keys=list(turn.keys), v=turn.value,
                             truth=turn.truth, prev=prev_used, note=S[b]["note"], raw=raw, value=val,
                             wellformed=wf, conf_inline=conf, lp=info["all"], lp_pre=info["pre"],
                             lp_fin=info["fin"], lp_fin_renorm_min=info["fin_renorm_lp_min"], tok_strs=info["tok_strs"], tok_lps=info["tok_lps"],
                             gen_tokens=len(co.token_ids)))

        # ---------- 2. gates (A1a / A1b / A2)
        if arm in GATE_ARMS:
            fire = []
            for b, r in enumerate(rows):
                reason = None
                if not r["wellformed"]:
                    reason = "format"
                elif arm == "A1b" and r["lp"]["lp_min"] < cfg["tau"]:
                    reason = "logit"
                elif arm == "A2" and S[b]["rng"].random() < cfg["rate"]:
                    reason = "random"
                r["gate"] = dict(fired=reason is not None, reason=reason)
                if reason:
                    fire.append(b)
            if fire:
                think = bool(cfg.get("recompute_think"))
                ver = cfg.get("recompute_version", 1)
                rp = []
                for b in fire:
                    u = PR.recompute_user(rows[b]["prev"], rows[b]["keys"], ver)
                    if think:
                        u += PR.LOW_EFFORT_SUFFIX
                    rp.append({"prompt_token_ids": sys_rec + ch.user(u) + ch.gen(think=think)})
                if think:
                    # low-effort thinking re-compute (pilot: 98.8% correct vs 5-24% without thinking);
                    # model-card sampling, seeded per (trajectory, turn)
                    a = cfg["audit_sampling"]
                    sps = [SamplingParams(temperature=a["temperature"], top_p=a["top_p"],
                                          max_tokens=cfg.get("recompute_budget", 1024), stop_token_ids=stop,
                                          seed=cfg.get("seed", 0) * 1_000_003 + rows[b]["sample_id"] * 10_007 + t + 7)
                           for b in fire]
                else:
                    sps = sp_rec
                routs = llm.generate(rp, sps, use_tqdm=False)
                for b, ro in zip(fire, routs):
                    full = ro.outputs[0].text
                    if think:
                        rraw = full.rsplit("</think>", 1)[1].strip() if "</think>" in full else ""
                    else:
                        rraw = full.strip()
                    rv, rwf, _ = parse_reply(rraw)
                    rows[b]["gate"].update(rec_raw=rraw, rec_value=rv, rec_wf=rwf,
                                           rec_tokens=len(ro.outputs[0].token_ids),
                                           rec_finished=(not think) or ("</think>" in full))

        # ---------- 3. commit + label
        for b, r in enumerate(rows):
            g = r.get("gate")
            if g and g["fired"] and g.get("rec_wf"):
                r["committed"], r["committed_wf"], r["hist_str"] = g["rec_value"], True, str(g["rec_value"])
            else:
                r["committed"], r["committed_wf"], r["hist_str"] = r["value"], r["wellformed"], r["raw"]
            r["kind"] = label_turn(r["committed"], r["committed_wf"], r["prev"], r["v"], r["truth"])
            r["retrieval_like"] = (retrieval_like(r["committed"], r["prev"], r["v"], dict_vals)
                                   if (K == 1 and r["kind"] == "numeric_slip") else None)
            S[b]["kinds"].append(r["kind"])

        # ---------- 4. forked probes (A0, A0_inline)
        if probes_on:
            bases = [S[b]["ctx"] + ch.user(user_txt[b]) + ch.asst(rows[b]["hist_str"]) for b in range(B)]
            reqs, tags = [], []
            for b in range(B):
                reqs += [({"prompt_token_ids": bases[b] + ch.user(PR.P_TRUE) + ch.gen()}, sp_lp1),
                         ({"prompt_token_ids": bases[b] + ch.user(PR.P_VERBAL) + ch.gen()}, sp_num)]
                tags += [(b, "ptrue"), (b, "verbal")]
            if t in checkpoints:
                nw = min(10, t + 1)
                n_wrong = [sum(k != "ok" for k in S[b]["kinds"][-nw:]) for b in range(B)]
                placebo = []
                for b in range(B):
                    pb = next((n_wrong[(b + j) % B] for j in range(1, B) if n_wrong[(b + j) % B] != n_wrong[b]),
                              n_wrong[(b + 1) % B])
                    placebo.append(pb)
                for b in range(B):
                    rows[b]["hc"] = dict(n_window=nw, n_true=n_wrong[b], n_placebo=placebo[b])
                    for cond, ev in [("none", ""), ("true", PR.evidence_line(n_wrong[b], nw)),
                                     ("placebo", PR.evidence_line(placebo[b], nw))]:
                        reqs.append(({"prompt_token_ids": bases[b] + ch.user(ev + PR.P_FORECAST) + ch.gen()}, sp_num))
                        tags.append((b, f"fore_{cond}"))
                    reqs.append(({"prompt_token_ids": bases[b] + ch.user(PR.P_AWARE) + ch.gen()}, sp_lp1))
                    tags.append((b, "aware"))
            pouts = llm.generate([q[0] for q in reqs], [q[1] for q in reqs], use_tqdm=False)
            for (b, k), po in zip(tags, pouts):
                co = po.outputs[0]
                pr = rows[b].setdefault("probes", {})
                if k == "ptrue":
                    pr[k] = p_from_first_token(co, yes_ids, no_ids)
                elif k == "aware":
                    lps = co.logprobs[0]
                    pa = [math.exp(lps[i].logprob) if i in lps else 0.0 for i in abc_ids]
                    s = sum(pa) or 1.0
                    pr[k] = [x / s for x in pa]
                else:
                    m = INT_RE.search(co.text)
                    pr[k] = int(m.group()) if m else None

        # ---------- 5. advance history / state
        for b, r in enumerate(rows):
            if arm in HISTORY_ARMS:
                S[b]["ctx"] = S[b]["ctx"] + ch.user(user_txt[b]) + ch.asst(r["hist_str"])
            S[b]["note"] = None
            S[b]["prev"] = next_prev(r["prev"] if arm != "D2" else S[b]["prev"], r["committed"])

        # ---------- 6. audits (A3 fresh / A4 in-context), every m turns
        if arm in AUDIT_ARMS and (t + 1) % cfg["m"] == 0:
            reqs = []
            for b in range(B):
                st = S[b]
                keys_since = [k for j in range(st["t0"] + 1, t + 1) for k in trajs[b].turns[j].keys]
                R = st["prev"]
                q = PR.audit_user(st["t0"], st["V"], keys_since, t, R)
                if cfg.get("audit_low_effort"):
                    q += PR.LOW_EFFORT_SUFFIX
                base = (sys_aud if arm == "A3" else st["ctx"])
                reqs.append({"prompt_token_ids": base + ch.user(q) + ch.gen(think=True)})
                rows[b]["audit"] = dict(t0=st["t0"], V=st["V"], R=R, n_keys=len(keys_since),
                                        actually_wrong=R != trajs[b].turns[t].truth)
            seeds = [cfg.get("seed", 0) * 1_000_003 + trajs[b].sample_id * 10_007 + t for b in range(B)]
            aouts = llm.generate(reqs, [sp_audit(s) for s in seeds], use_tqdm=False)
            for b, ao in enumerate(aouts):
                text = ao.outputs[0].text
                pa = parse_audit(text)
                n_think = (len(tok.encode(text.split("</think>")[0], add_special_tokens=False))
                           if "</think>" in text else len(ao.outputs[0].token_ids))
                a = rows[b]["audit"]
                a.update(pa, think_tokens=n_think, gen_tokens=len(ao.outputs[0].token_ids))
                st = S[b]
                if pa["finished"] and pa["verdict"] == "CORRECT":
                    st["V"], st["t0"] = a["R"], t
                    a["action"] = "accept"
                elif pa["finished"] and pa["verdict"] == "WRONG" and pa["total"] is not None:
                    st["note"] = (t, pa["total"])
                    st["prev"] = pa["total"]
                    st["V"], st["t0"] = pa["total"], t
                    a["action"] = "correct"
                else:
                    a["action"] = "none"   # unfinished, or WRONG without a total
                a["auditor_total_correct"] = (pa["total"] == trajs[b].turns[t].truth) if pa["total"] is not None else None

        for b, r in enumerate(rows):
            fo.write(json.dumps(r) + "\n")
            S[b]["n_gate"] += int(bool(r.get("gate", {}).get("fired")))
            if "audit" in r:
                S[b]["n_audit"] += 1
                S[b]["n_correct"] += int(r["audit"]["action"] == "correct")
        fo.flush()
        if t % 10 == 0 or t == T - 1:
            acc = sum(r["kind"] == "ok" for r in rows) / B
            msg = (f"[{label}] turn {t} elapsed {time.time() - t_start:.0f}s ctx_len {len(S[0]['ctx'])} "
                   f"acc_now {acc:.2f}")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

    from .metrics import turns_to_first_error
    summaries = []
    for b, tr in enumerate(trajs):
        k = S[b]["kinds"]
        summaries.append(dict(sample_id=tr.sample_id, arm=arm, T=len(k), accuracy=sum(x == "ok" for x in k) / len(k),
                              turns_to_first_error=turns_to_first_error(k),
                              kinds={x: k.count(x) for x in sorted(set(k))},
                              gate_fired=S[b]["n_gate"], audits=S[b]["n_audit"], corrections=S[b]["n_correct"],
                              finished_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    return summaries


def run(cfg, trajs, dictionary, llm, tok, out_dir, resume=False):
    """Run an arm over `trajs` in groups of cfg['group_size'] (default: all at once).

    Saving: every turn's rows are flushed to rows.jsonl; every finished trajectory gets a line in
    trajectories.jsonl immediately. Resume (resume=True, same out_dir): trajectories already in
    trajectories.jsonl are skipped, and rows of unfinished trajectories are discarded first.
    """
    out_dir = Path(out_dir)
    done = set()
    if resume and (out_dir / "trajectories.jsonl").exists():
        with open(out_dir / "trajectories.jsonl") as f:
            done = {json.loads(l)["sample_id"] for l in f if l.strip()}
        keep = []
        if (out_dir / "rows.jsonl").exists():
            with open(out_dir / "rows.jsonl") as f:
                keep = [l for l in f if l.strip() and json.loads(l)["sample_id"] in done]
        with open(out_dir / "rows.jsonl", "w") as f:
            f.writelines(keep)
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    todo = [tr for tr in trajs if tr.sample_id not in done]
    gs = cfg.get("group_size") or len(todo) or 1
    fo = open(out_dir / "rows.jsonl", "a")
    ft = open(out_dir / "trajectories.jsonl", "a")
    log = open(out_dir / "run.log", "a")
    t_start = time.time()
    if done:
        log.write(f"RESUME: skipping {len(done)} finished trajectories\n")
    for g0 in range(0, len(todo), gs):
        group = todo[g0:g0 + gs]
        label = f"traj {group[0].sample_id}..{group[-1].sample_id}"
        for s in _run_group(cfg, group, dictionary, llm, tok, fo, log, t_start, label):
            ft.write(json.dumps(s) + "\n")
            ft.flush()
            log.write(f"saved trajectory {s['sample_id']} accuracy={s['accuracy']:.3f}\n")
            log.flush()
    fo.close()
    ft.close()
    log.write(f"DONE {time.time() - t_start:.0f}s\n")
    log.close()
