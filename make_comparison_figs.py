#!/usr/bin/env python3
"""
보고서용 Marabou vs alpha,beta-CROWN 비교 figure 생성.
서버(~/trust_ai/assignment4)에서 실행:
    python make_comparison_figs.py
출력: report_figs/ 아래에 PNG 4개
"""
import os, re, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
ABC_RESULTS = os.path.expanduser("~/trust_ai/assignment4/results")
MARABOU_JSON = os.path.expanduser(
    "~/trust_ai/marabou-neural-verification/results/summary.json")
OUT_DIR = os.path.expanduser("~/trust_ai/assignment4/report_figs")
os.makedirs(OUT_DIR, exist_ok=True)

TIMEOUT_THRESH = 290.0   # Marabou: 이 시간 이상이면 timeout으로 간주 (budget 300s)

# ---------------------------------------------------------------------------
# 1. alpha,beta-CROWN 결과 파싱 (results_summary.txt)
# ---------------------------------------------------------------------------
def parse_abcrown_summary(path):
    """results_summary.txt 끝의 집계 블록에서 수치 추출."""
    txt = open(path).read()
    def grab(key):
        m = re.search(rf"{key}\s*:\s*([\d.]+)", txt)
        return float(m.group(1)) if m else None
    return {
        "verified":  int(grab("verified")),
        "falsified": int(grab("falsified")),
        "timeout":   int(grab("timeout")),
        "mean_time": grab("mean time"),
        "max_time":  grab("max time"),
    }

def load_abcrown(prefix):
    """prefix='mnist_fc_eps_' 또는 'fashion_mnist_eps_' → {eps: stats} 반환."""
    out = {}
    for d in sorted(glob.glob(os.path.join(ABC_RESULTS, prefix + "*"))):
        name = os.path.basename(d)
        summ = os.path.join(d, "results_summary.txt")
        if not os.path.isfile(summ):
            continue
        if prefix == "mnist_fc_eps_":
            eps = float(name.replace(prefix, ""))           # 0.05
        else:
            frac = name.replace(prefix, "").replace("_255", "")  # '8'
            eps = int(frac) / 255.0
        out[eps] = parse_abcrown_summary(summ)
    return out

abc_mnist = load_abcrown("mnist_fc_eps_")
abc_fashion = load_abcrown("fashion_mnist_eps_")

# ---------------------------------------------------------------------------
# 2. Marabou 결과 파싱 (summary.json) → eps별 집계
# ---------------------------------------------------------------------------
mara = json.load(open(MARABOU_JSON))
# eps별로 묶기
from collections import defaultdict
mara_by_eps = defaultdict(list)
for rec in mara:
    mara_by_eps[round(float(rec["epsilon"]), 3)].append(rec)

def marabou_stats(recs):
    n = len(recs)
    # UNSAT = robust(verified), SAT = 반례(falsified)
    unsat = sum(1 for r in recs if r["result"] == "UNSAT")
    sat   = sum(1 for r in recs if r["result"] == "SAT")
    times = [float(r["time"]) for r in recs]
    timeouts = sum(1 for t in times if t >= TIMEOUT_THRESH)
    mean_t = sum(times) / n
    max_t = max(times)
    return dict(n=n, unsat=unsat, sat=sat, timeout=timeouts,
                mean_time=mean_t, max_time=max_t)

mara_eps = sorted(mara_by_eps.keys())
mara_agg = {e: marabou_stats(mara_by_eps[e]) for e in mara_eps}

# 공통 eps (둘 다 측정한 값) — MNIST FC 비교용
common_eps = sorted(set(abc_mnist.keys()) & set(mara_agg.keys()))
print("alpha,beta-CROWN MNIST eps:", sorted(abc_mnist.keys()))
print("Marabou eps:", mara_eps)
print("common eps:", common_eps)

# ---------------------------------------------------------------------------
# Figure A: robustness 경계 비교 (verified% / UNSAT% vs eps)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.8))
abc_pct = [abc_mnist[e]["verified"] / 50 * 100 for e in common_eps]
mar_pct = [mara_agg[e]["unsat"] / mara_agg[e]["n"] * 100 for e in common_eps]
ax.plot(common_eps, abc_pct, "o-", color="#2980B9", lw=2,
        label="α,β-CROWN  (verified %)")
ax.plot(common_eps, mar_pct, "s--", color="#C0392B", lw=2,
        label="Marabou  (UNSAT %)")
ax.axhline(50, color="grey", ls=":", lw=1)
ax.set_xlabel("perturbation radius  ε  (L∞, raw pixel)")
ax.set_ylabel("robust fraction (%)")
ax.set_title("Robustness boundary: same MNIST FC, two verifiers", fontsize=10)
ax.set_ylim(-3, 103)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cmp_boundary.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Figure B: 검증 시간 비교 (mean & max vs eps, 로그 스케일)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.8))
abc_mean = [abc_mnist[e]["mean_time"] for e in common_eps]
abc_max  = [abc_mnist[e]["max_time"]  for e in common_eps]
mar_mean = [mara_agg[e]["mean_time"]  for e in common_eps]
mar_max  = [mara_agg[e]["max_time"]   for e in common_eps]
ax.plot(common_eps, mar_max,  "s--", color="#C0392B", lw=2, label="Marabou max")
ax.plot(common_eps, mar_mean, "s-",  color="#E59866", lw=2, label="Marabou mean")
ax.plot(common_eps, abc_max,  "o--", color="#2980B9", lw=2, label="α,β-CROWN max")
ax.plot(common_eps, abc_mean, "o-",  color="#7FB3D5", lw=2, label="α,β-CROWN mean")
ax.set_yscale("log")
ax.set_xlabel("perturbation radius  ε  (L∞, raw pixel)")
ax.set_ylabel("verification time per query (s, log scale)")
ax.set_title("Verification time: same MNIST FC, two verifiers", fontsize=10)
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cmp_time.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Figure C: FashionMNIST 결과 (막대 + 런타임) — 재생성
# ---------------------------------------------------------------------------
fm_eps_int = sorted(int(round(e * 255)) for e in abc_fashion.keys())
fm_keys = sorted(abc_fashion.keys())
fm_ver = [abc_fashion[e]["verified"]  for e in fm_keys]
fm_fal = [abc_fashion[e]["falsified"] for e in fm_keys]
fm_to  = [abc_fashion[e]["timeout"]   for e in fm_keys]
fm_tm  = [abc_fashion[e]["mean_time"] for e in fm_keys]

fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
xs = range(len(fm_keys))
ax1.bar(xs, fm_ver, label="verified", color="#4C9A2A")
ax1.bar(xs, fm_fal, bottom=fm_ver, label="falsified", color="#C0392B")
bottom2 = [v + f for v, f in zip(fm_ver, fm_fal)]
ax1.bar(xs, fm_to, bottom=bottom2, label="timeout", color="#7F8C8D")
ax1.set_xticks(list(xs))
ax1.set_xticklabels([f"{e}/255" for e in fm_eps_int])
ax1.set_xlabel("perturbation radius  ε (L∞)")
ax1.set_ylabel("instances (of 50)")
ax1.set_ylim(0, 55)
ax2 = ax1.twinx()
ax2.plot(xs, fm_tm, "o--", color="#2C3E50", label="mean time (s)")
ax2.set_ylabel("mean time (s)")
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper center", fontsize=7, ncol=4)
ax1.set_title("FashionMNIST CNN: outcomes and runtime vs ε", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fashion_outcomes.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Figure D: 두 모델 verified% vs eps (alpha,beta-CROWN)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
mn_keys = sorted(abc_mnist.keys())
ax.plot(mn_keys, [abc_mnist[e]["verified"] / 50 * 100 for e in mn_keys],
        "s-", color="#2980B9", label="MNIST FC")
ax.plot(fm_keys, [abc_fashion[e]["verified"] / 50 * 100 for e in fm_keys],
        "o-", color="#E67E22", label="FashionMNIST CNN")
ax.set_xlabel("perturbation radius  ε  (L∞, raw pixel)")
ax.set_ylabel("verified accuracy (%)")
ax.set_title("α,β-CROWN verified accuracy vs ε (both models)", fontsize=10)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "abc_both_models.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 콘솔 요약 (보고서 표에 그대로 쓸 수 있게)
# ---------------------------------------------------------------------------
print("\n=== MNIST FC: Marabou vs alpha,beta-CROWN ===")
print(f"{'eps':>6} | {'Mara UNSAT%':>11} {'Mara mean':>9} {'Mara max':>8} "
      f"{'Mara TO':>7} | {'ABC ver%':>8} {'ABC mean':>8} {'ABC max':>7} {'ABC TO':>6}")
for e in common_eps:
    m, a = mara_agg[e], abc_mnist[e]
    print(f"{e:>6.3f} | {m['unsat']/m['n']*100:>10.0f}% {m['mean_time']:>9.2f} "
          f"{m['max_time']:>8.2f} {m['timeout']:>7d} | "
          f"{a['verified']/50*100:>7.0f}% {a['mean_time']:>8.2f} "
          f"{a['max_time']:>7.2f} {a['timeout']:>6d}")

print("\nfigures saved to:", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", f)