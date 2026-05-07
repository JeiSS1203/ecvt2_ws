#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


JOINTS = ("UPJ5", "UPJ6")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.05,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})


def read_rows(csv_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def extract_series(rows: List[Dict[str, float]]) -> Dict[str, List[float]]:
    if not rows:
        raise RuntimeError("CSV has no data rows")

    t0 = rows[0].get("sim_time_sec", rows[0].get("ros_time_sec", 0.0))
    series = {
        "time": [],
        "trajectory_elapsed": [],
    }
    for joint in JOINTS:
        series[f"{joint}_position"] = []
        series[f"{joint}_velocity"] = []

    for row in rows:
        t = row.get("sim_time_sec", row.get("ros_time_sec", 0.0)) - t0
        series["time"].append(t)
        series["trajectory_elapsed"].append(row.get("trajectory_elapsed_sec", t))
        for joint in JOINTS:
            series[f"{joint}_position"].append(row[f"{joint}_position"])
            series[f"{joint}_velocity"].append(row[f"{joint}_velocity"])

    return series


def infer_terminal_time(series: Dict[str, List[float]], terminal_time: Optional[float]) -> float:
    if terminal_time is not None:
        return terminal_time
    return max(series["trajectory_elapsed"])


def mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def rms(values: List[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / max(1, len(values)))


def tail_indices(time: List[float], start_time: float) -> List[int]:
    return [i for i, t in enumerate(time) if t >= start_time]


def estimate_equilibrium(
    time: List[float],
    position: List[float],
    tail_window_sec: float,
) -> float:
    end_time = time[-1]
    start_time = max(time[0], end_time - tail_window_sec)
    idx = tail_indices(time, start_time)
    if not idx:
        return position[-1]
    return mean([position[i] for i in idx])


def first_settled_time(
    time: List[float],
    values: List[float],
    threshold: float,
    terminal_time: float,
    hold_sec: float,
) -> Optional[float]:
    start_idx = next((i for i, t in enumerate(time) if t >= terminal_time), None)
    if start_idx is None:
        return None

    for i in range(start_idx, len(time)):
        t0 = time[i]
        if abs(values[i]) > threshold:
            continue
        window = [abs(values[j]) for j, t in enumerate(time) if t0 <= t <= t0 + hold_sec]
        if window and max(window) <= threshold:
            return t0 - terminal_time
    return None


def padded_limits(values: List[float], pad_ratio: float = 0.22, min_span: float = 1e-5) -> Tuple[float, float]:
    v_min = min(values)
    v_max = max(values)
    span = max(v_max - v_min, min_span)
    center = 0.5 * (v_min + v_max)
    half = 0.5 * span * (1.0 + 2.0 * pad_ratio)
    return center - half, center + half


def style_axis(ax):
    ax.minorticks_on()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(which="major", length=3.5, width=0.7)
    ax.tick_params(which="minor", length=2.0, width=0.55)
    ax.grid(True, which="major", linewidth=0.45, alpha=0.28)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.13)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def make_summary(
    series: Dict[str, List[float]],
    terminal_time: float,
    velocity_threshold: float,
    position_threshold: float,
    hold_sec: float,
    eq_window_sec: float,
) -> Tuple[Dict, Dict[str, float]]:
    time = series["time"]
    summary = {
        "terminal_time_sec": terminal_time,
        "velocity_threshold_rad_s": velocity_threshold,
        "position_threshold_rad": position_threshold,
        "settle_hold_sec": hold_sec,
        "equilibrium_estimation_window_sec": eq_window_sec,
        "joints": {},
    }
    eq = {}

    for joint in JOINTS:
        pos = series[f"{joint}_position"]
        vel = series[f"{joint}_velocity"]
        eq_pos = estimate_equilibrium(time, pos, eq_window_sec)
        eq[joint] = eq_pos

        post_idx = tail_indices(time, terminal_time)
        post_vel = [vel[i] for i in post_idx]
        post_err = [pos[i] - eq_pos for i in post_idx]

        summary["joints"][joint] = {
            "estimated_equilibrium_position_rad": eq_pos,
            "terminal_position_rad": pos[post_idx[0]] if post_idx else pos[-1],
            "terminal_velocity_rad_s": vel[post_idx[0]] if post_idx else vel[-1],
            "post_terminal_rms_velocity_rad_s": rms(post_vel),
            "post_terminal_max_abs_velocity_rad_s": max([abs(v) for v in post_vel], default=float("nan")),
            "post_terminal_rms_position_error_rad": rms(post_err),
            "post_terminal_max_abs_position_error_rad": max([abs(e) for e in post_err], default=float("nan")),
            "velocity_settling_time_after_terminal_sec": first_settled_time(
                time, vel, velocity_threshold, terminal_time, hold_sec
            ),
            "position_settling_time_after_terminal_sec": first_settled_time(
                time, [p - eq_pos for p in pos], position_threshold, terminal_time, hold_sec
            ),
        }

    return summary, eq


def plot_paper_figure(
    series: Dict[str, List[float]],
    eq: Dict[str, float],
    terminal_time: float,
    output_png: str,
):
    time = series["time"]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.65), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.93, hspace=0.24, wspace=0.30)

    colors = {
        "UPJ5": "#1f77b4",
        "UPJ6": "#d62728",
    }

    for col, joint in enumerate(JOINTS):
        pos = series[f"{joint}_position"]
        vel = series[f"{joint}_velocity"]
        pos_err = [p - eq[joint] for p in pos]

        ax_pos = axes[0][col]
        ax_vel = axes[1][col]

        ax_pos.plot(time, pos_err, color=colors[joint], linewidth=1.05)
        ax_pos.axhline(0.0, color="0.15", linewidth=0.75, linestyle="--", alpha=0.8)
        ax_pos.axvline(terminal_time, color="0.15", linewidth=0.8, linestyle=":", alpha=0.9)
        ax_pos.set_title(f"({chr(ord('a') + col)}) {joint} position error", fontsize=10)
        ax_pos.set_ylabel(r"$q_P - q_{P,eq}$ [rad]")
        ax_pos.set_ylim(*padded_limits(pos_err))
        style_axis(ax_pos)

        ax_vel.plot(time, vel, color=colors[joint], linewidth=1.05)
        ax_vel.axhline(0.0, color="0.15", linewidth=0.75, linestyle="--", alpha=0.8)
        ax_vel.axvline(terminal_time, color="0.15", linewidth=0.8, linestyle=":", alpha=0.9)
        ax_vel.set_title(f"({chr(ord('c') + col)}) {joint} velocity damping", fontsize=10)
        ax_vel.set_xlabel("Time after trajectory start [s]")
        ax_vel.set_ylabel(r"$\dot{q}_P$ [rad/s]")
        ax_vel.set_ylim(*padded_limits(vel))
        style_axis(ax_vel)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paper-style plots from upj5_upj6_joint_states.csv"
    )
    parser.add_argument("--csv", default="/home/jin/harco/ecvt2_ws/upj5_upj6_joint_states.csv")
    parser.add_argument("--output-dir", default="/home/jin/harco/ecvt2_ws/passive_joint_figures")
    parser.add_argument("--prefix", default="passive_terminal")
    parser.add_argument("--terminal-time", type=float, default=None)
    parser.add_argument("--velocity-threshold", type=float, default=0.01)
    parser.add_argument("--position-threshold", type=float, default=0.005)
    parser.add_argument("--settle-hold-sec", type=float, default=0.3)
    parser.add_argument("--eq-window-sec", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.csv)
    series = extract_series(rows)
    terminal_time = infer_terminal_time(series, args.terminal_time)
    summary, eq = make_summary(
        series,
        terminal_time,
        args.velocity_threshold,
        args.position_threshold,
        args.settle_hold_sec,
        args.eq_window_sec,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    paper_png = os.path.join(args.output_dir, f"{args.prefix}_paper.png")
    summary_json = os.path.join(args.output_dir, f"{args.prefix}_summary.json")

    plot_paper_figure(
        series,
        eq,
        terminal_time,
        paper_png,
    )
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved paper plot: {paper_png}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
