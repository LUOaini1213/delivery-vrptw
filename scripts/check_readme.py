"""Generate the README result tables from results/*.json, or check that they match.

    python scripts/check_readme.py --write   # regenerate every <!-- BEGIN:x --> ... <!-- END:x --> block
    python scripts/check_readme.py --check   # exit 1 if any block differs from what the JSON gives

Numbers in the README are only ever written through this script.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

import numpy as np
from common import RESULTS, ROOT

README = ROOT / "README.md"
FAMS = ["C1", "C2", "R1", "R2", "RC1", "RC2"]


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x, nd=2):
    v = round(float(x), nd)
    return f"{v + 0.0:.{nd}f}%"  # +0.0 turns -0.0 into 0.0


def f2(x, nd=2):
    return f"{x:.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- stage 1a

def small_rows(d):
    rows = []
    for n in sorted({r["n"] for r in d["runs"]}):
        for fam in FAMS + ["all"]:
            rs = [r for r in d["runs"] if r["n"] == n and (fam == "all" or r["family"] == fam)]
            if not rs:
                continue
            opt = [r for r in rs if r["mip"]["optimal"]]
            hit_any = hit_all = 0
            gaps, i1g = [], []
            for r in opt:
                o = r["mip"]["objective"]
                g = [(a["distance"] / o - 1) * 100 for a in r["alns"]]
                hit_any += min(g) <= 1e-6
                hit_all += max(g) <= 1e-6
                gaps.append(np.mean(g))
                i1g.append((r["i1"]["distance"] / o - 1) * 100)
            nonopt = [r for r in rs if not r["mip"]["optimal"]]
            better = sum(min(a["distance"] for a in r["alns"]) <= r["mip"].get("objective", np.inf) + 1e-6
                         for r in nonopt)
            rows.append([n, fam if fam != "all" else "**全部**", len(rs), len(opt),
                         f"{hit_any}/{len(opt)}", f"{hit_all}/{len(opt)}",
                         pct(np.mean(gaps), 3) if gaps else "-", pct(np.mean(i1g)) if i1g else "-",
                         f"{better}/{len(nonopt)}" if nonopt else "-",
                         f2(np.mean([r["mip"]["seconds"] for r in rs]), 1),
                         f2(np.mean([a["seconds"] for r in rs for a in r["alns"]]), 1)])
    return rows


def block_small(d):
    hdr = ["n", "族", "实例", "MIP 证明最优", "ALNS 命中最优（5 种子任一）", "ALNS 命中（5 种子全部）",
           "ALNS 平均 gap", "I1 平均 gap", "MIP 未证最优时 ALNS ≤ MIP 现解", "MIP 平均秒", "ALNS 平均秒/次"]
    lim = ", ".join(f"n={k}: {v:.0f}s" for k, v in d["mip_time_limit"].items())
    it = ", ".join(f"n={k}: {v}" for k, v in d["alns_iters"].items())
    note = (f"MIP 时限 {lim}；ALNS 迭代 {it}，种子 {d['seeds']}；gap 只在 MIP 证明最优的实例上计算。"
            f"并行 {d['workers']} 进程，总墙钟 {d['wall_seconds'] / 60:.0f} 分钟。")
    return table(hdr, small_rows(d)) + "\n\n" + note


# ----------------------------------------------------------------------------- stage 1b

def block_small_findings(d):
    lines = []
    for n in sorted({r["n"] for r in d["runs"]}):
        rs = [r for r in d["runs"] if r["n"] == n]
        opt = [r for r in rs if r["mip"]["optimal"]]
        hit = sum(min(a["distance"] for a in r["alns"]) <= r["mip"]["objective"] + 1e-6 for r in opt)
        runs_hit = sum(a["distance"] <= r["mip"]["objective"] + 1e-6 for r in opt for a in r["alns"])
        runs_all = sum(len(r["alns"]) for r in opt)
        worst = max(((max(a["distance"] for a in r["alns"]) / r["mip"]["objective"] - 1) * 100, r["instance"])
                    for r in opt) if opt else None
        nonopt = [r for r in rs if not r["mip"]["optimal"]]
        strictly = [r["instance"] for r in nonopt
                    if min(a["distance"] for a in r["alns"]) < r["mip"].get("objective", np.inf) - 1e-6]
        s = (f"- n = {n}：MIP 在时限内证明最优 {len(opt)}/{len(rs)} 个；其中 ALNS 至少一个种子命中最优 {hit}/{len(opt)}，"
             f"单次运行命中 {runs_hit}/{runs_all}")
        if worst:
            s += f"，最差的单次运行差 {worst[0]:.2f}%（{worst[1]}）"
        s += "。"
        if nonopt:
            gaps = [r["mip"]["gap"] * 100 for r in nonopt if r["mip"].get("gap") is not None]
            s += (f" 未证明最优的 {len(nonopt)} 个（族：{', '.join(sorted({r['family'] for r in nonopt}))}）里，"
                  f"ALNS 严格好于 MIP 时限内现解的有 {len(strictly)} 个；这些实例 MIP 剩余相对 gap 中位数 "
                  f"{np.median(gaps):.1f}%。")
        lines.append(s)
    lines.append("- 注意：MIP 最优是“已证明”的数字，ALNS 的数字是启发式结果；上表把两者分开统计。")
    return "\n".join(lines)


def best_of(runs):
    return min(runs, key=lambda r: (r["alns"]["vehicles"], r["alns"]["distance"]))


def solomon_by_instance(d):
    by = {}
    for r in d["runs"]:
        by.setdefault(r["instance"], []).append(r)
    return by


def solomon_rows(d):
    by = solomon_by_instance(d)
    rows = []
    for fam in FAMS + ["all"]:
        names = [n for n in by if fam == "all" or by[n][0]["family"] == fam]
        if not names:
            continue
        bks_v = sum(by[n][0]["bks"]["vehicles"] for n in names)
        bks_d = sum(by[n][0]["bks"]["distance"] for n in names)
        best = {n: best_of(by[n]) for n in names}
        v_best = sum(best[n]["alns"]["vehicles"] for n in names)
        d_best = sum(best[n]["alns"]["distance"] for n in names)
        v_mean = sum(np.mean([r["alns"]["vehicles"] for r in by[n]]) for n in names)
        same = [n for n in names if best[n]["alns"]["vehicles"] == best[n]["bks"]["vehicles"]]
        # BKS distances are published rounded to 2 decimals, so round ours the same way first
        gap_same = [(round(best[n]["alns"]["distance"], 2) / best[n]["bks"]["distance"] - 1) * 100 for n in same]
        gap_all_seeds = [(round(r["alns"]["distance"], 2) / r["bks"]["distance"] - 1) * 100
                         for n in names for r in by[n] if r["alns"]["vehicles"] == r["bks"]["vehicles"]]
        n_runs_same = len(gap_all_seeds)
        i1_v = sum(by[n][0]["i1"]["vehicles"] for n in names)
        rows.append([fam if fam != "all" else "**全部**", len(names), bks_v, v_best, f2(v_mean, 1), i1_v,
                     f"{len(same)}/{len(names)}",
                     pct(np.mean(gap_same)) if gap_same else "-",
                     f"{pct(np.mean(gap_all_seeds))} ({n_runs_same} 次)" if gap_all_seeds else "-",
                     f2(bks_d, 0), f2(d_best, 0),
                     f2(np.mean([r["alns"]["seconds"] for n in names for r in by[n]]), 0)])
    return rows


def block_solomon(d):
    hdr = ["族", "实例", "BKS 车辆和", "ALNS 车辆和（3 种子取最好）", "ALNS 车辆和（种子平均）", "I1 车辆和",
           "车辆数=BKS 的实例", "这些实例上距离 gap（最好）", "距离 gap（所有车辆数=BKS 的单次运行）",
           "BKS 总距离", "ALNS 总距离（最好）", "平均秒/次"]
    note = (f"预算：每次 {d['iters']} 次 ALNS 迭代（其中至多 70% 用于减车阶段），种子 {d['seeds']}，"
            f"{d['workers']} 进程并行（秒数含并行争用），总墙钟 {d['wall_seconds'] / 60:.0f} 分钟。"
            "“最好”= 3 个种子里按（车辆数，距离）字典序最好的一次。距离 gap 只在车辆数与 BKS 相同时有意义；"
            "车辆数更多时距离常常反而更短，不能直接比。")
    return table(hdr, solomon_rows(d)) + "\n\n" + note


def block_solomon_detail(d):
    by = solomon_by_instance(d)
    rows = []
    for n in sorted(by, key=lambda x: (FAMS.index(by[x][0]["family"]), x)):
        b = best_of(by[n])
        bk = b["bks"]
        seeds = "; ".join(f"{r['alns']['vehicles']}/{r['alns']['distance']:.2f}" for r in sorted(by[n], key=lambda r: r["seed"]))
        gap = (pct((round(b['alns']['distance'], 2) / bk['distance'] - 1) * 100)
               if b["alns"]["vehicles"] == bk["vehicles"] else f"+{b['alns']['vehicles'] - bk['vehicles']} 车")
        rows.append([n, f"{bk['vehicles']}/{bk['distance']:.2f} ({bk['ref']})",
                     f"{by[n][0]['i1']['vehicles']}/{by[n][0]['i1']['distance']:.2f}", seeds, gap])
    return ("<details><summary>逐实例（车辆数/距离）</summary>\n\n"
            + table(["实例", "BKS（出处代码）", "I1", "ALNS 种子 " + "/".join(str(s) for s in d["seeds"]), "最好解 vs BKS"], rows)
            + "\n\n</details>")


def block_solomon_findings(d):
    by = solomon_by_instance(d)
    best = {n: best_of(by[n]) for n in by}
    exact_all = [n for n in by if all(r["alns"]["vehicles"] == r["bks"]["vehicles"]
                                      and abs(round(r["alns"]["distance"], 2) - r["bks"]["distance"]) < 0.006
                                      for r in by[n])]
    extra = sorted(n for n in by if best[n]["alns"]["vehicles"] > best[n]["bks"]["vehicles"])
    fewer = sorted(n for n in by if best[n]["alns"]["vehicles"] < best[n]["bks"]["vehicles"])
    gaps = sorted(((round(best[n]["alns"]["distance"], 2) / best[n]["bks"]["distance"] - 1) * 100, n)
                  for n in by if best[n]["alns"]["vehicles"] == best[n]["bks"]["vehicles"])
    worst = ", ".join(f"{n} {g:.2f}%" for g, n in gaps[-3:][::-1])
    i1 = sum(by[n][0]["i1"]["vehicles"] for n in by)
    v = sum(best[n]["alns"]["vehicles"] for n in by)
    bv = sum(best[n]["bks"]["vehicles"] for n in by)
    c_names = [n for n in by if n.startswith("C")]
    lines = [
        f"- C1/C2 两族 {sum(n in exact_all for n in c_names)}/{len(c_names)} 个实例三个种子都回到 BKS（车辆数与距离到两位小数）。",
        f"- 车辆数：ALNS 最好解合计 {v} 辆，BKS {bv} 辆，I1 {i1} 辆；比 BKS 多 1 辆以上的实例 {len(extra)} 个"
        f"（{', '.join(extra)}），分布在 {', '.join(sorted({by[n][0]['family'] for n in extra}))} 族。"
        + ("" if not fewer else f" 少于 BKS：{fewer}（需核查）"),
        f"- 距离：车辆数与 BKS 相同的 {len(gaps)} 个实例上，最好解平均差 {np.mean([g for g, _ in gaps]):.2f}%，"
        f"差得最多的是 {worst}。",
        f"- 结论：在每次 {d['iters']} 次迭代的预算下，聚类型（C）实例能稳定回到 BKS；随机/混合型（R/RC）实例离文献最好解"
        "还有百分之几的距离差和个别车辆差，这不是 state-of-the-art 的求解器。",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- stage 2

def block_drone_small(d):
    s = d["small"]
    rows = []
    for n in sorted({r["n"] for r in s["runs"]}):
        rs = [r for r in s["runs"] if r["n"] == n]
        opt = [r for r in rs if r["mip"]["optimal"]]
        gaps = [np.mean([(a["objective"] / r["mip"]["objective"] - 1) * 100 for a in r["alns"]]) for r in opt]
        any_hit = sum(min(a["objective"] for a in r["alns"]) <= r["mip"]["objective"] * (1 + 1e-6) for r in opt)
        all_hit = sum(max(a["objective"] for a in r["alns"]) <= r["mip"]["objective"] * (1 + 1e-6) for r in opt)
        i1 = [(r["i1"] / r["mip"]["objective"] - 1) * 100 for r in opt]
        rows.append([n, len(rs), len(opt), f"{any_hit}/{len(opt)}", f"{all_hit}/{len(opt)}",
                     pct(np.mean(gaps), 3) if gaps else "-", pct(np.mean(i1)) if i1 else "-",
                     f2(np.mean([r["mip"]["seconds"] for r in rs]), 1)])
    note = (f"目标：总能耗 Wh。实例：`generate(seed, n, depots, horizon={s['horizon']:.0f})`，"
            f"奇数种子 1 个场站、偶数种子 2 个场站；MIP 时限 {s['mip_time_limit']:.0f}s；"
            f"ALNS {s['alns_iters']} 次迭代 × 种子 {s['alns_seeds']}。")
    return table(["n", "实例", "MIP 证明最优", "ALNS 命中（任一种子）", "ALNS 命中（全部种子）", "ALNS 平均 gap",
                  "I1 平均 gap", "MIP 平均秒"], rows) + "\n\n" + note


def block_drone_large(d):
    L = d["large"]
    rows = []
    by = {}
    for r in L["runs"]:
        by.setdefault(r["instance_seed"], []).append(r)
    tot = []
    for s in sorted(by):
        rs = by[s]
        i1 = rs[0]["i1"]
        e = np.mean([r["alns"]["energy_wh"] for r in rs])
        tot.append((i1["energy_wh"], e))
        rows.append([rs[0]["name"], f2(i1["energy_wh"], 0), i1["trips"], i1["drones"],
                     f2(e, 0), f2(np.mean([r["alns"]["trips"] for r in rs]), 1),
                     f2(np.mean([r["alns"]["drones"] for r in rs]), 1),
                     pct((e / i1["energy_wh"] - 1) * 100, 1),
                     f2(np.mean([r["alns"]["seconds"] for r in rs]), 0)])
    red = np.mean([(e / i - 1) * 100 for i, e in tot])
    note = (f"ALNS：{L['alns_iters']} 次迭代 × 种子 {L['alns_seeds']}（表中取平均）。无人机数来自贪心的航次串联"
            f"（换电 {d['spec']['swap_min']:.0f} 分钟），不是联合优化的结果。5 个实例平均能耗变化 {red:.1f}%。")
    return table(["实例", "I1 能耗 Wh", "I1 航次", "I1 无人机", "ALNS 能耗 Wh", "ALNS 航次", "ALNS 无人机",
                  "能耗变化", "ALNS 秒/次"], rows) + "\n\n" + note


def block_drone_findings(d):
    s = d["small"]["runs"]
    opt = [r for r in s if r["mip"]["optimal"]]
    hit = sum(min(a["objective"] for a in r["alns"]) <= r["mip"]["objective"] * (1 + 1e-6) for r in opt)
    L = d["large"]["runs"]
    red = [(r["alns"]["energy_wh"] / r["i1"]["energy_wh"] - 1) * 100 for r in L]
    dd = [r["alns"]["drones"] - r["i1"]["drones"] for r in L]
    lines = [
        f"- 小实例：MIP 证明最优 {len(opt)}/{len(s)} 个，ALNS（任一种子）命中其中 {hit} 个。能耗、载重、最长时长、多场站"
        "都在 MIP 与 ALNS 里同时建模，且 MIP 在 ≤ 6 顾客的随机实例上与穷举一致（测试）。",
        f"- 60 单实例：ALNS 相对 I1 的能耗变化在 {min(red):.1f}% 到 {max(red):.1f}% 之间（{len(L)} 次运行）。"
        f"串联后的无人机数变化在 {min(dd):+d} 到 {max(dd):+d} 架之间——ALNS 只优化能耗，机队规模不在目标里，"
        "所以省电的计划不一定少用无人机。",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- stage 3

LABEL = {"greedy": "(a) I1 静态", "alns": "(b) ALNS 静态", "robust15": "(c) ALNS + 15% 缓冲",
         "robust30": "(c') ALNS + 30% 缓冲", "rolling": "(d) 滚动重规划"}


def ci(v, scale=1.0, nd=2):
    m, lo, hi = (x * scale for x in v)
    lo, hi = min(lo, hi), max(lo, hi)
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def sim_table(summ, base="alns"):
    rows = []
    for p, v in summ.items():
        dv = v.get("diff_vs_" + base)
        rows.append([LABEL[p], v["n"], pct(v["on_time"] * 100, 1), f2(v["mean_late"], 3), f2(v["p95_late"], 2),
                     f2(v["reserve_viol"], 2), f2(v["depleted"], 3), f2(v["energy"] / 1000, 2),
                     f2(v["drones"], 1), f2(v["trips"], 1),
                     ci(dv["on_time"], 100, 1) if dv else "基准",
                     ci(dv["reserve_viol"]) if dv else "基准",
                     ci(dv["energy"], 1 / 1000) if dv else "基准"])
    return table(["策略", "样本", "准时率", "平均迟到 min", "P95 迟到 min", "低于备用电量航次/天", "电量耗尽航次/天",
                  "实际能耗 kWh", "无人机", "航次", "Δ准时率 pp [95% CI]", "Δ低备电航次 [95% CI]",
                  "Δ能耗 kWh [95% CI]"], rows)


def block_sim_static(d):
    note = (f"{len(d['instances'])} 个实例 × 每实例 {d['n_static']} 个场景 = {sum(v['n'] for v in [d['summary_static']['alns']])} "
            f"个配对样本；Δ 为相对 (b) 的逐场景配对差，均值的 95% 百分位 bootstrap 区间（2000 次重抽样）。"
            f"规划：ALNS {d['plan_iters']} 次迭代，种子 {d['plan_seed']}。")
    return sim_table(d["summary_static"]) + "\n\n" + note


def block_sim_rolling(d):
    note = (f"滚动策略较慢（平均每场景 {d['rolling_seconds_mean']:.0f} 秒），只在每个实例的前 {d['n_rolling']} 个场景上跑，"
            f"因此这张表的所有策略都只用这 {d['summary_with_rolling']['alns']['n']} 个配对样本；每次重规划 ALNS "
            f"{d['rolling_iters_per_replan']} 次迭代。")
    return sim_table(d["summary_with_rolling"]) + "\n\n" + note


def paired_raw(rows, a, b, m):
    from vrptw.sim import bootstrap_ci
    return bootstrap_ci([r[a][m] - r[b][m] for r in rows if a in r and b in r])


def paired(rows, a, b, m, scale=1.0, nd=2):
    return ci(paired_raw(rows, a, b, m), scale, nd)


def block_sim_findings(d):
    S = d["summary_static"]
    sub = [r for r in d["rows"] if "rolling" in r]
    g, r15, r30 = (S[k]["diff_vs_alns"] for k in ("greedy", "robust15", "robust30"))
    lines = [
        f"- 同样静态执行，ALNS 计划相对 I1 计划：能耗 {ci(g['energy'], -1 / 1000)} kWh/天（负号 = ALNS 更省），"
        f"低于备用电量的航次 {ci(g['reserve_viol'], -1)} 次/天，准时率 {ci(g['on_time'], -100, 1)} pp。"
        "两者执行方式完全相同，差别只来自计划本身。",
        f"- 缓冲换准时：15% / 30% 缓冲把准时率提高 {ci(r15['on_time'], 100, 1)} / {ci(r30['on_time'], 100, 1)} pp，"
        f"代价是能耗 {ci(r15['energy'], 1 / 1000)} / {ci(r30['energy'], 1 / 1000)} kWh/天、"
        f"无人机从 {S['alns']['drones']:.1f} 到 {S['robust15']['drones']:.1f} / {S['robust30']['drones']:.1f} 架（5 个实例平均）。"
        f"30% 缓冲下仍有 {S['robust30']['reserve_viol']:.2f} 次/天落地时低于备用电量，缓冲没有把它消除。",
        f"- 滚动重规划 vs 30% 静态缓冲（同 {len(sub)} 个配对场景）：准时率 "
        f"{paired(sub, 'rolling', 'robust30', 'on_time', 100, 1)} pp，能耗 {paired(sub, 'rolling', 'robust30', 'energy', 1 / 1000)} kWh/天，"
        f"无人机 {paired(sub, 'rolling', 'robust30', 'drones')} 架，航次 {paired(sub, 'rolling', 'robust30', 'trips', 1, 1)}，"
        f"低备电航次 {paired(sub, 'rolling', 'robust30', 'reserve_viol')} 次/天。",
    ]
    ot = paired_raw(sub, "rolling", "robust30", "on_time")
    if ot[1] < 0 < ot[2]:
        lines.append("  准时率差异的 95% CI 跨过 0：在这个仿真设定里，每小时按观测风速重规划并没有比一开始就留 30% 缓冲更准时；"
                     "它的差别主要在资源侧（上面的无人机、航次、能耗）。")
    return "\n".join(lines)


def block_en_findings(_):
    sm, so, dr, si = (load(x) for x in ("small_mip_vs_alns.json", "solomon100.json", "drone.json", "simulation.json"))
    out = []
    if sm:
        opt = [r for r in sm["runs"] if r["mip"]["optimal"]]
        hit = sum(min(a["distance"] for a in r["alns"]) <= r["mip"]["objective"] + 1e-6 for r in opt)
        out.append(f"- Small Solomon subsets (first 10/25 customers, distance objective): HiGHS proved {len(opt)}/"
                   f"{len(sm['runs'])} optima; ALNS matched {hit}/{len(opt)} of them with at least one of "
                   f"{len(sm['seeds'])} seeds.")
    if so:
        by = solomon_by_instance(so)
        best = {n: best_of(by[n]) for n in by}
        same = [n for n in by if best[n]["alns"]["vehicles"] == best[n]["bks"]["vehicles"]]
        g = np.mean([(round(best[n]["alns"]["distance"], 2) / best[n]["bks"]["distance"] - 1) * 100 for n in same])
        out.append(f"- 56 Solomon 100-customer instances, {so['iters']} ALNS iterations x {len(so['seeds'])} seeds: "
                   f"{sum(best[n]['alns']['vehicles'] for n in by)} vehicles in total vs "
                   f"{sum(best[n]['bks']['vehicles'] for n in by)} for the best-known solutions; on the {len(same)} "
                   f"instances with the BKS fleet size the best run is {g:.2f}% longer on average.")
    if dr:
        L = dr["large"]["runs"]
        red = np.mean([(r["alns"]["energy_wh"] / r["i1"]["energy_wh"] - 1) * 100 for r in L])
        out.append(f"- Drone variant (60 orders, 2 depots): ALNS changes planned energy by {red:.1f}% on average vs "
                   "Solomon I1 (fleet size is not in its objective).")
    if si:
        S = si["summary_static"]
        sub = [r for r in si["rows"] if "rolling" in r]
        ot = paired_raw(sub, "rolling", "robust30", "on_time")
        out.append(f"- Simulation ({S['alns']['n']} paired scenarios): on-time rate {S['greedy']['on_time'] * 100:.1f}% (I1 plan), "
                   f"{S['alns']['on_time'] * 100:.1f}% (ALNS plan), {S['robust15']['on_time'] * 100:.1f}% / "
                   f"{S['robust30']['on_time'] * 100:.1f}% (ALNS with 15% / 30% buffers). Hourly rolling re-planning vs the "
                   f"30% buffer: {ot[0] * 100:+.1f} pp on-time (95% CI {ot[1] * 100:+.1f} to {ot[2] * 100:+.1f}), i.e. "
                   + ("no clear gain." if ot[1] < 0 < ot[2] else "see tables."))
    return "\n".join(out)


# ----------------------------------------------------------------------------- misc

def block_mutation(d):
    return (f"变异测试：{d['killed']}/{d['total']} 个故意植入的缺陷被测试抓到（详见 [MUTATION.md](MUTATION.md)）；"
            f"未修改副本测试通过、等价变异存活（证明工具能报告“漏网”）。"
            + ("首轮漏网的变异、以及为它们补上的测试也记在 MUTATION.md 里。" if d.get("history") else ""))


def count_tests():
    p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", p.stdout)
    if not m:
        raise SystemExit("could not count tests:\n" + p.stdout[-2000:])
    return int(m.group(1))


def block_tests(_):
    return f"`python -m pytest -q` 共 {count_tests()} 个测试。"


def block_machine(_):
    seen = []
    for name in ["small_mip_vs_alns.json", "solomon100.json", "drone.json", "simulation.json"]:
        d = load(name)
        if d:
            m = d["machine"]
            seen.append(f"- `{name}`：{m['cpu']}，{m['logical_cpus']} 逻辑核，{m['os']}，Python {m['python']}，"
                        f"numpy {m['numpy']}，scipy {m['scipy']}（HiGHS 经 `scipy.optimize.milp`）")
    return "\n".join(seen)


BLOCKS = {
    "small": ("small_mip_vs_alns.json", block_small),
    "small_findings": ("small_mip_vs_alns.json", block_small_findings),
    "solomon": ("solomon100.json", block_solomon),
    "solomon_detail": ("solomon100.json", block_solomon_detail),
    "solomon_findings": ("solomon100.json", block_solomon_findings),
    "drone_small": ("drone.json", block_drone_small),
    "drone_large": ("drone.json", block_drone_large),
    "drone_findings": ("drone.json", block_drone_findings),
    "sim_static": ("simulation.json", block_sim_static),
    "sim_rolling": ("simulation.json", block_sim_rolling),
    "sim_findings": ("simulation.json", block_sim_findings),
    "mutation": ("mutation.json", block_mutation),
    "tests": (None, block_tests),
    "en_findings": (None, block_en_findings),
    "machine": (None, block_machine),
}


def render(text, allow_missing=False):
    found = []
    for key, (src, fn) in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN:{key} -->\n)(.*?)(<!-- END:{key} -->)", re.S)
        m = pat.search(text)
        if not m:
            continue
        found.append(key)
        data = load(src) if src else None
        if src and data is None:
            if allow_missing:
                print(f"warning: block {key} skipped, results/{src} missing")
                continue
            raise SystemExit(f"README block {key} needs results/{src}, which is missing")
        text = text[: m.start()] + m.group(1) + fn(data) + "\n" + m.group(3) + text[m.end():]
    return text, found


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cur = README.read_text(encoding="utf-8")
    new, found = render(cur, allow_missing=a.write)
    missing = [k for k in BLOCKS if k not in found]
    if a.write:
        with open(README, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(new)
        print(f"README updated ({len(found)} blocks); blocks not present: {missing}")
        return
    if missing:
        print(f"README is missing generated blocks: {missing}")
        sys.exit(1)
    if new != cur:
        import difflib
        diff = difflib.unified_diff(cur.splitlines(), new.splitlines(), "README.md", "generated", lineterm="", n=0)
        print("\n".join(list(diff)[:60]))
        print("README tables do not match results/*.json - run: python scripts/check_readme.py --write")
        sys.exit(1)
    print(f"README matches results ({len(found)} generated blocks)")


if __name__ == "__main__":
    main()
