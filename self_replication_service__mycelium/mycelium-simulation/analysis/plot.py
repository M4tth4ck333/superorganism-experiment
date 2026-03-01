#!/usr/bin/env python3
"""Plot simulation results from event CSV."""

import argparse
import csv
import os
import sys
from collections import defaultdict
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_detail(detail: str) -> dict[str, str]:
    result = {}
    for token in detail.split():
        if "=" in token:
            k, _, v = token.partition("=")
            result[k] = v
    return result


def gini(balances: list[int]) -> float:
    xs = sorted(b for b in balances if b > 0)
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total == 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * weighted) / (n * total) - (n + 1) / n


def _moving_average(values: list[float], window: int) -> list[float]:
    return [
        sum(values[i : i + window]) / window
        for i in range(len(values) - window + 1)
    ]


def extract_summaries(events: list[dict]) -> dict:
    ticks = []
    living_counts = []
    total_balances = []

    for ev in events:
        if ev["event"] != "summary":
            continue
        tick = int(ev["tick"])
        parts = parse_detail(ev["detail"])
        ticks.append(tick)
        living_counts.append(int(parts.get("living", 0)))
        total_balances.append(int(parts.get("total_bal", 0)))

    return {"ticks": ticks, "living": living_counts, "total_balance": total_balances}


def extract_spawn_death_counts(events: list[dict]) -> dict:
    spawns: dict[int, int] = {}
    deaths: dict[int, int] = {}

    for ev in events:
        tick = int(ev["tick"])
        if ev["event"] == "spawn":
            spawns[tick] = spawns.get(tick, 0) + 1
        elif ev["event"] == "death":
            deaths[tick] = deaths.get(tick, 0) + 1

    return {"spawns": spawns, "deaths": deaths}


def extract_balance_history(events: list[dict]) -> dict:
    balances: dict[str, int] = {}
    history: dict[str, list] = defaultdict(list)
    snapshots: dict[int, dict] = {}

    for ev in events:
        tick = int(ev["tick"])
        node_id = ev["node_id"]
        event = ev["event"]
        parts = parse_detail(ev["detail"])

        if event == "spawn":
            bal = int(parts.get("balance", 0))
            balances[node_id] = bal
            history[node_id].append((tick, bal))

        elif event == "rent":
            remaining = int(parts.get("remaining", 0))
            balances[node_id] = remaining
            history[node_id].append((tick, remaining))

        elif event == "faucet":
            amount = int(parts.get("amount", 0))
            balances[node_id] = balances.get(node_id, 0) + amount
            history[node_id].append((tick, balances[node_id]))

        elif event == "inheritance":
            amount = int(parts.get("amount", 0))
            sender = parts.get("from", "")
            balances[node_id] = balances.get(node_id, 0) + amount
            history[node_id].append((tick, balances[node_id]))
            if sender and sender in balances:
                balances[sender] -= amount
                history[sender].append((tick, balances[sender]))

        elif event == "death":
            history[node_id].append((tick, 0))
            balances.pop(node_id, None)

        elif event == "summary":
            snapshots[tick] = dict(balances)

    return {"history": dict(history), "snapshots": snapshots}


def extract_lifespans(events: list[dict]) -> list[dict]:
    births: dict[str, int] = {}
    dead: set[str] = set()
    lifespans: list[dict] = []
    final_tick = 0

    for ev in events:
        tick = int(ev["tick"])
        node_id = ev["node_id"]
        if tick > final_tick:
            final_tick = tick

        if ev["event"] == "spawn":
            births[node_id] = tick
        elif ev["event"] == "death":
            parts = parse_detail(ev["detail"])
            cause = parts.get("reason", "unknown")
            birth_tick = births.get(node_id, 0)
            lifespans.append({
                "node_id": node_id,
                "birth_tick": birth_tick,
                "death_tick": tick,
                "lifespan": tick - birth_tick,
                "cause": cause,
            })
            dead.add(node_id)

    for node_id, birth_tick in births.items():
        if node_id not in dead:
            lifespans.append({
                "node_id": node_id,
                "birth_tick": birth_tick,
                "death_tick": final_tick,
                "lifespan": final_tick - birth_tick,
                "cause": "alive",
            })

    return lifespans


def plot_population(summaries: dict, output_dir: str) -> None:
    ticks = summaries["ticks"]
    if not ticks:
        print("No summary data — skipping population.png")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(ticks, summaries["living"], color="green", linewidth=1.5, label="Living nodes")
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("Living Nodes", color="green")
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.set_title("Mycelium Colony Population Over Time")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ticks, summaries["total_balance"], color="orange", linewidth=1, alpha=0.7, label="Total BTC")
    ax2.set_ylabel("Total Satoshis in Economy", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    fig.tight_layout()
    path = os.path.join(output_dir, "population.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_spawn_death(spawn_deaths: dict, output_dir: str) -> None:
    all_ticks = sorted(set(list(spawn_deaths["spawns"].keys()) + list(spawn_deaths["deaths"].keys())))
    if not all_ticks:
        print("No spawn/death data — skipping spawn_death.png")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    spawn_vals = [spawn_deaths["spawns"].get(t, 0) for t in all_ticks]
    death_vals = [spawn_deaths["deaths"].get(t, 0) for t in all_ticks]
    ax.bar(all_ticks, spawn_vals, color="green", alpha=0.6, label="Spawns", width=0.8)
    ax.bar(all_ticks, [-d for d in death_vals], color="red", alpha=0.6, label="Deaths", width=0.8)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Count")
    ax.set_title("Spawn / Death Events Per Tick")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "spawn_death.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_gini(balance_hist: dict, output_dir: str) -> None:
    snapshots = balance_hist["snapshots"]
    if not snapshots:
        print("No balance snapshots — skipping gini.png")
        return

    ticks = sorted(snapshots.keys())
    gini_vals = [gini(list(snapshots[t].values())) for t in ticks]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ticks, gini_vals, color="purple", linewidth=1.2, alpha=0.5, label="Gini coefficient")

    if len(gini_vals) >= 10:
        window = max(5, len(gini_vals) // 20)
        smoothed = _moving_average(gini_vals, window)
        smooth_ticks = ticks[window - 1:]
        ax.plot(smooth_ticks, smoothed, color="purple", linewidth=2.0, label=f"Smoothed (w={window})")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Wealth Inequality Over Time (Gini Coefficient)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "gini.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_wealth_flows(events: list[dict], output_dir: str) -> None:
    tick_data: dict[int, dict] = defaultdict(lambda: defaultdict(int))

    for ev in events:
        tick = int(ev["tick"])
        parts = parse_detail(ev["detail"])
        if ev["event"] == "faucet":
            tick_data[tick]["faucet"] += int(parts.get("amount", 0))
        elif ev["event"] == "rent":
            tick_data[tick]["rent"] += int(parts.get("amount", 0))
        elif ev["event"] == "inheritance":
            reason = parts.get("reason", "")
            amount = int(parts.get("amount", 0))
            if reason in ("failsafe", "bankrupt_sweep", "failsafe_sweep"):
                tick_data[tick]["redistribution"] += amount
            else:
                tick_data[tick]["spawn_inh"] += amount

    all_ticks = sorted(tick_data.keys())
    if not all_ticks:
        print("No flow data — skipping wealth_flows.png")
        return

    cum_faucet, cum_rent, cum_spawn, cum_redis = [], [], [], []
    tf = tr = ts = tx = 0
    for t in all_ticks:
        tf += tick_data[t]["faucet"]
        tr += tick_data[t]["rent"]
        ts += tick_data[t]["spawn_inh"]
        tx += tick_data[t]["redistribution"]
        cum_faucet.append(tf)
        cum_rent.append(tr)
        cum_spawn.append(ts)
        cum_redis.append(tx)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_ticks, cum_faucet, color="gold", linewidth=2, label="Faucet injected")
    ax.plot(all_ticks, cum_rent, color="red", linewidth=2, label="Rent collected")
    ax.plot(all_ticks, cum_spawn, color="green", linewidth=2, label="Spawn inheritance")
    ax.plot(all_ticks, cum_redis, color="blue", linewidth=2, label="Failsafe/sweep transfers")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Cumulative Satoshis")
    ax.set_title("Cumulative Economic Flows")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "wealth_flows.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_behavior_mix(events: list[dict], summaries: dict, output_dir: str) -> None:
    ticks = summaries["ticks"]
    if not ticks:
        print("No summary data — skipping behavior_mix.png")
        return

    living_by_tick = dict(zip(summaries["ticks"], summaries["living"]))
    spawns_per_tick: dict[int, int] = defaultdict(int)
    failsafe_per_tick: dict[int, int] = defaultdict(int)

    for ev in events:
        tick = int(ev["tick"])
        if ev["event"] == "spawn":
            spawns_per_tick[tick] += 1
        elif ev["event"] == "death":
            parts = parse_detail(ev["detail"])
            if parts.get("reason") == "failsafe":
                failsafe_per_tick[tick] += 1

    spawn_fracs, failsafe_fracs, none_fracs = [], [], []
    for t in ticks:
        living = living_by_tick.get(t, 0)
        s = spawns_per_tick.get(t, 0)
        f = failsafe_per_tick.get(t, 0)
        total = max(1, living + f)  # end-of-tick living already excludes failsafe dead
        none_count = max(0, total - s - f)
        denom = s + f + none_count or 1
        spawn_fracs.append(s / denom)
        failsafe_fracs.append(f / denom)
        none_fracs.append(none_count / denom)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        ticks,
        none_fracs, spawn_fracs, failsafe_fracs,
        labels=["None (accumulate)", "Spawn", "Failsafe"],
        colors=["#aaaaaa", "#4caf50", "#f44336"],
        alpha=0.8,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Fraction of Living Nodes")
    ax.set_title("Behavioral Composition Per Tick")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(output_dir, "behavior_mix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_lifespan_distribution(lifespans: list[dict], output_dir: str) -> None:
    bankrupt = [ls["lifespan"] for ls in lifespans if ls["cause"] == "bankrupt"]
    failsafe = [ls["lifespan"] for ls in lifespans if ls["cause"] == "failsafe"]

    if not bankrupt and not failsafe:
        print("No lifespan data — skipping lifespan.png")
        return

    max_life = max(
        (max(bankrupt) if bankrupt else 0),
        (max(failsafe) if failsafe else 0),
    )
    bins = min(40, max_life + 1) if max_life > 0 else 10

    fig, ax = plt.subplots(figsize=(10, 5))
    if bankrupt:
        ax.hist(bankrupt, bins=bins, color="red", alpha=0.6,
                label=f"Bankrupt (n={len(bankrupt)})", density=True)
    if failsafe:
        ax.hist(failsafe, bins=bins, color="blue", alpha=0.6,
                label=f"Failsafe (n={len(failsafe)})", density=True)
    ax.set_xlabel("Lifespan (ticks)")
    ax.set_ylabel("Density")
    ax.set_title("Lifespan Distribution by Death Cause")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "lifespan.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_survival_curve(lifespans: list[dict], output_dir: str) -> None:
    cause_colors = {"bankrupt": "red", "failsafe": "blue"}
    fig, ax = plt.subplots(figsize=(10, 5))
    any_data = False

    for cause, color in cause_colors.items():
        group = [ls for ls in lifespans if ls["cause"] == cause]
        if not group:
            continue
        any_data = True
        n = len(group)
        death_times = sorted(ls["lifespan"] for ls in group)

        xs = [0]
        ys = [1.0]
        surviving = n
        for t in death_times:
            xs.append(t)
            ys.append(surviving / n)
            surviving -= 1
            xs.append(t)
            ys.append(surviving / n)

        ax.plot(xs, ys, color=color, linewidth=2, label=f"{cause} (n={n})")

    if not any_data:
        print("No death data — skipping survival_curve.png")
        plt.close(fig)
        return

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Ticks Since Birth")
    ax.set_ylabel("Fraction Surviving")
    ax.set_title("Survival Curves by Death Cause")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "survival_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_lineage_depth(events: list[dict], output_dir: str) -> None:
    parent_of: dict[str, str] = {}
    birth_tick: dict[str, int] = {}

    for ev in events:
        if ev["event"] != "spawn":
            continue
        node_id = ev["node_id"]
        parts = parse_detail(ev["detail"])
        birth_tick[node_id] = int(ev["tick"])
        parent = parts.get("parent", "")
        if parent:
            parent_of[node_id] = parent

    if not birth_tick:
        print("No spawn data — skipping lineage_depth.png")
        return

    generation: dict[str, int] = {}

    def get_gen(start: str) -> int:
        if start in generation:
            return generation[start]
        chain = [start]
        while chain[-1] in parent_of and parent_of[chain[-1]] not in generation:
            chain.append(parent_of[chain[-1]])
        base = chain[-1]
        generation[base] = 0 if base not in parent_of else generation[parent_of[base]] + 1
        for nid in reversed(chain[:-1]):
            generation[nid] = generation[parent_of[nid]] + 1
        return generation[start]

    for nid in birth_tick:
        get_gen(nid)

    tick_gens: dict[int, list[int]] = defaultdict(list)
    for nid, g in generation.items():
        tick_gens[birth_tick[nid]].append(g)

    all_ticks = sorted(tick_gens.keys())
    max_gens = [max(tick_gens[t]) for t in all_ticks]
    mean_gens = [sum(tick_gens[t]) / len(tick_gens[t]) for t in all_ticks]

    scatter_x = [birth_tick[n] for n in birth_tick]
    scatter_y = [generation[n] for n in birth_tick]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.scatter(scatter_x, scatter_y, alpha=0.5, s=20, color="steelblue")
    ax1.set_xlabel("Birth Tick")
    ax1.set_ylabel("Generation")
    ax1.set_title("Node Generation at Birth")
    ax1.grid(True, alpha=0.3)

    ax2.plot(all_ticks, max_gens, color="darkblue", linewidth=1.5, label="Max generation born")
    ax2.plot(all_ticks, mean_gens, color="cornflowerblue", linewidth=1.5,
             linestyle="--", label="Mean generation born")
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("Generation Depth")
    ax2.set_title("Generational Depth Per Tick")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "lineage_depth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_caution_evolution(events: list[dict], output_dir: str) -> None:
    birth_ticks = []
    birth_cautions = []
    for ev in events:
        if ev["event"] != "spawn":
            continue
        parts = parse_detail(ev["detail"])
        if "caution" not in parts:
            continue
        birth_ticks.append(int(ev["tick"]))
        birth_cautions.append(float(parts["caution"]))

    summary_ticks = []
    avg_cautions = []
    min_cautions = []
    max_cautions = []
    for ev in events:
        if ev["event"] != "trait_summary":
            continue
        parts = parse_detail(ev["detail"])
        summary_ticks.append(int(ev["tick"]))
        avg_cautions.append(float(parts.get("avg_caution", 0)))
        min_cautions.append(float(parts.get("min_caution", 0)))
        max_cautions.append(float(parts.get("max_caution", 0)))

    if not birth_ticks and not summary_ticks:
        print("No caution data — skipping caution_evolution.png")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    if birth_ticks:
        ax1.scatter(birth_ticks, birth_cautions, alpha=0.5, s=20, color="teal")
    ax1.set_xlabel("Birth Tick")
    ax1.set_ylabel("Caution as % increase from baseline minimum")
    ax1.set_title("Per-Node Caution at Birth")
    ax1.grid(True, alpha=0.3)

    if summary_ticks:
        ax2.plot(summary_ticks, avg_cautions, color="teal", linewidth=2, label="Mean")
        ax2.fill_between(summary_ticks, min_cautions, max_cautions,
                         color="teal", alpha=0.2, label="Min–Max")
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("Caution as % increase from baseline minimum")
    ax2.set_title("Population Caution Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "caution_evolution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_failsafe_effectiveness(events: list[dict], lifespans: list[dict], output_dir: str) -> None:
    lifespan_by_node = {ls["node_id"]: ls for ls in lifespans}

    spawned_at: dict[str, int] = {}
    for ev in events:
        if ev["event"] == "spawn":
            parts = parse_detail(ev["detail"])
            parent = parts.get("parent", "")
            if parent and parent not in spawned_at:
                spawned_at[parent] = int(ev["tick"])

    donations = []
    for ev in events:
        if ev["event"] != "inheritance":
            continue
        parts = parse_detail(ev["detail"])
        if parts.get("reason") != "failsafe":
            continue
        donations.append({
            "recipient": ev["node_id"],
            "amount": int(parts.get("amount", 0)),
            "tick": int(ev["tick"]),
        })

    if not donations:
        print("No failsafe donations — skipping failsafe_effectiveness.png")
        return

    xs, ys, colors = [], [], []
    for d in donations:
        recipient = d["recipient"]
        if recipient not in lifespan_by_node:
            continue
        ls = lifespan_by_node[recipient]
        ticks_after = max(0, ls["death_tick"] - d["tick"])
        did_spawn = recipient in spawned_at and spawned_at[recipient] > d["tick"]

        xs.append(d["amount"])
        ys.append(ticks_after)
        if ls["cause"] == "alive":
            colors.append("gold")
        elif did_spawn:
            colors.append("green")
        elif ticks_after > 3:
            colors.append("yellow")
        else:
            colors.append("red")

    if not xs:
        print("No valid failsafe data — skipping failsafe_effectiveness.png")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, c=colors, alpha=0.7, s=60, edgecolors="k", linewidths=0.5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
               markersize=10, label="Recipient spawned after"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
               markersize=10, label="Survived >3 ticks"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=10, label="Died within 3 ticks"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",
               markersize=10, label="Still alive"),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("Donated Satoshis")
    ax.set_ylabel("Ticks Recipient Survived After Donation")
    ax.set_title("Failsafe Effectiveness: Cooperation Outcomes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "failsafe_effectiveness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_wealth_trajectories(balance_hist: dict, lifespans: list[dict], output_dir: str) -> None:
    history = balance_hist["history"]
    if not history:
        print("No balance history — skipping wealth_trajectories.png")
        return

    cause_by_node = {ls["node_id"]: ls["cause"] for ls in lifespans}
    cause_colors = {"bankrupt": "red", "failsafe": "blue", "alive": "green"}
    n_nodes = len(history)

    fig, ax = plt.subplots(figsize=(14, 7))

    if n_nodes <= 50:
        for node_id, points in history.items():
            if len(points) < 2:
                continue
            ticks_h = [p[0] for p in points]
            bals_h = [p[1] for p in points]
            cause = cause_by_node.get(node_id, "alive")
            color = cause_colors.get(cause, "gray")
            ax.plot(ticks_h, bals_h, color=color, alpha=0.3, linewidth=0.8)

        legend_elements = [
            Line2D([0], [0], color=c, linewidth=2, label=cause)
            for cause, c in cause_colors.items()
        ]
        ax.legend(handles=legend_elements)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Balance (satoshis)")
        ax.set_title("Per-Node Balance Trajectories")

    else:
        lifespan_by_node = {ls["node_id"]: ls for ls in lifespans}
        all_x, all_y = [], []
        for node_id, points in history.items():
            if len(points) < 2:
                continue
            ls_info = lifespan_by_node.get(node_id)
            if not ls_info:
                continue
            birth = ls_info["birth_tick"]
            death = ls_info["death_tick"]
            span = death - birth
            if span == 0:
                continue
            for tick_h, bal in points:
                all_x.append((tick_h - birth) / span)
                all_y.append(bal)

        if all_x:
            ax.hist2d(all_x, all_y, bins=50, cmap="hot_r")
        ax.set_xlabel("Relative Age (0=birth, 1=death)")
        ax.set_ylabel("Balance (satoshis)")
        ax.set_title(f"Balance Distribution Heatmap (n={n_nodes} nodes)")

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(output_dir, "wealth_trajectories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_wealth_histogram(balance_hist: dict, output_dir: str) -> None:
    snapshots = balance_hist["snapshots"]
    if not snapshots:
        print("No snapshots — skipping wealth_histogram.png")
        return

    ticks = sorted(snapshots.keys())
    n = len(ticks)
    if n < 4:
        slice_ticks = ticks
    else:
        indices = [n // 4, n // 2, 3 * n // 4, n - 1]
        slice_ticks = [ticks[i] for i in indices]

    fig, axes = plt.subplots(1, len(slice_ticks), figsize=(4 * len(slice_ticks), 5))
    if len(slice_ticks) == 1:
        axes = [axes]

    for ax, t in zip(axes, slice_ticks):
        bals = [v for v in snapshots[t].values() if v > 0]
        if bals:
            ax.hist(bals, bins=20, color="steelblue", alpha=0.8, edgecolor="k")
        ax.set_title(f"Tick {t}")
        ax.set_xlabel("Balance (sat)")
        ax.set_ylabel("Nodes")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Wealth Distribution at Time Slices", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "wealth_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_all(events: list[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    summaries = extract_summaries(events)
    spawn_deaths = extract_spawn_death_counts(events)
    balance_hist = extract_balance_history(events)
    lifespans = extract_lifespans(events)

    plot_population(summaries, output_dir)
    plot_spawn_death(spawn_deaths, output_dir)
    plot_gini(balance_hist, output_dir)
    plot_wealth_flows(events, output_dir)
    plot_behavior_mix(events, summaries, output_dir)
    plot_lifespan_distribution(lifespans, output_dir)
    plot_survival_curve(lifespans, output_dir)
    plot_lineage_depth(events, output_dir)
    plot_caution_evolution(events, output_dir)
    plot_failsafe_effectiveness(events, lifespans, output_dir)
    plot_wealth_trajectories(balance_hist, lifespans, output_dir)
    plot_wealth_histogram(balance_hist, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot mycelium simulation results")
    parser.add_argument("events_csv", nargs="?", default="data/events.csv", help="Path to events CSV")
    parser.add_argument("-o", "--output", default="data/plots", help="Output directory for plots")
    args = parser.parse_args()

    if not os.path.isfile(args.events_csv):
        print(f"Events file not found: {args.events_csv}")
        sys.exit(1)

    events = load_events(args.events_csv)
    plot_all(events, args.output)


if __name__ == "__main__":
    main()
