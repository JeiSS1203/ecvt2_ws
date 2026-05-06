#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


PASSIVE_JOINTS = ("UPJ5", "UPJ6")


def point_time_sec(point) -> float:
    return float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
          rows.append({key: float(value) for key, value in row.items() if value != ""})
    return rows


def read_meta(csv_path: str) -> Dict:
    meta_path = os.path.splitext(csv_path)[0] + ".json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def sample_at(rows: List[Dict[str, float]], t_query: float, key: str) -> float:
    if not rows:
        return float("nan")
    best = min(rows, key=lambda r: abs(r["t"] - t_query))
    return best.get(key, float("nan"))


def post_terminal_stats(rows: List[Dict[str, float]], terminal_time: Optional[float]) -> Dict[str, float]:
    if terminal_time is None or not rows:
        return {}

    tail = [r for r in rows if r["t"] >= terminal_time]
    if not tail:
        return {}

    out: Dict[str, float] = {}
    for joint in PASSIVE_JOINTS:
        pos_key = f"{joint}_pos"
        vel_key = f"{joint}_vel"
        pos = [r[pos_key] for r in tail if pos_key in r]
        vel = [r[vel_key] for r in tail if vel_key in r]
        if pos:
            out[f"{joint}_terminal_pos"] = sample_at(rows, terminal_time, pos_key)
            out[f"{joint}_post_mean_pos"] = sum(pos) / len(pos)
            out[f"{joint}_post_peak_to_peak_pos"] = max(pos) - min(pos)
        if vel:
            out[f"{joint}_terminal_vel"] = sample_at(rows, terminal_time, vel_key)
            out[f"{joint}_post_rms_vel"] = math.sqrt(sum(v * v for v in vel) / len(vel))
            out[f"{joint}_post_max_abs_vel"] = max(abs(v) for v in vel)
    return out


def make_plot(
    datasets: List[Tuple[str, List[Dict[str, float]], Dict]],
    output_png: str,
    target_upj5: Optional[float] = None,
    target_upj6: Optional[float] = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8), sharex=True)
    fig.subplots_adjust(hspace=0.18, wspace=0.22)

    targets = {"UPJ5": target_upj5, "UPJ6": target_upj6}

    for label, rows, meta in datasets:
        if not rows:
            continue

        t = [r["t"] for r in rows]
        for col, joint in enumerate(PASSIVE_JOINTS):
            pos = [r[f"{joint}_pos"] for r in rows]
            vel = [r[f"{joint}_vel"] for r in rows]

            axes[0][col].plot(t, pos, linewidth=1.8, label=label)
            axes[1][col].plot(t, vel, linewidth=1.8, label=label)

            terminal_time = meta.get("terminal_time")
            if terminal_time is not None:
                axes[0][col].axvline(terminal_time, color="k", linestyle="--", linewidth=1.0, alpha=0.55)
                axes[1][col].axvline(terminal_time, color="k", linestyle="--", linewidth=1.0, alpha=0.55)

            if targets[joint] is not None:
                axes[0][col].axhline(targets[joint], color="tab:red", linestyle=":", linewidth=1.2, alpha=0.75)

    for col, joint in enumerate(PASSIVE_JOINTS):
        axes[0][col].set_title(f"{joint} position")
        axes[1][col].set_title(f"{joint} velocity")
        axes[1][col].set_xlabel("Time [s]")
        axes[0][col].set_ylabel("Position [rad]")
        axes[1][col].set_ylabel("Velocity [rad/s]")
        axes[0][col].grid(True, alpha=0.25)
        axes[1][col].grid(True, alpha=0.25)
        axes[0][col].legend(loc="best")

    fig.suptitle("Passive joint terminal behavior")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


class PassiveJointRecorder(Node):
    def __init__(self, args):
        super().__init__("passive_joint_plot_recorder")
        self.args = args
        self.label = args.label
        self.rows: List[Dict[str, float]] = []
        self.t0: Optional[float] = None
        self.terminal_time: Optional[float] = args.terminal_time
        self.last_sample_wall = time.monotonic()

        self.create_subscription(JointState, args.joint_state_topic, self.joint_state_callback, 50)
        self.create_subscription(JointTrajectory, args.trajectory_topic, self.trajectory_callback, 10)

        self.get_logger().warn(
            f"Recording {PASSIVE_JOINTS} from {args.joint_state_topic}; "
            f"trajectory_topic={args.trajectory_topic}; duration={args.duration}s"
        )

    def trajectory_callback(self, msg: JointTrajectory):
        if not msg.points:
            return
        self.t0 = time.monotonic()
        if self.args.terminal_time is None:
            self.terminal_time = point_time_sec(msg.points[-1])
        self.rows = []
        self.get_logger().warn(
            f"Trajectory received. Reset recording t=0. terminal_time={self.terminal_time:.4f}s"
        )

    def joint_state_callback(self, msg: JointState):
        if self.t0 is None:
            if self.args.start_without_trajectory:
                self.t0 = time.monotonic()
            else:
                return

        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        if any(joint not in name_to_idx for joint in PASSIVE_JOINTS):
            return

        now = time.monotonic()
        t = now - self.t0
        row = {"t": t}
        for joint in PASSIVE_JOINTS:
            idx = name_to_idx[joint]
            row[f"{joint}_pos"] = float(msg.position[idx])
            row[f"{joint}_vel"] = float(msg.velocity[idx]) if idx < len(msg.velocity) else float("nan")
        self.rows.append(row)
        self.last_sample_wall = now

    def done(self) -> bool:
        if self.t0 is None:
            return False
        return (time.monotonic() - self.t0) >= self.args.duration

    def save(self):
        ensure_dir(self.args.output_dir)
        stem = self.args.output_prefix
        csv_path = os.path.join(self.args.output_dir, f"{stem}.csv")
        meta_path = os.path.join(self.args.output_dir, f"{stem}.json")
        png_path = os.path.join(self.args.output_dir, f"{stem}.png")

        fieldnames = ["t", "UPJ5_pos", "UPJ5_vel", "UPJ6_pos", "UPJ6_vel"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        meta = {
            "label": self.label,
            "joint_state_topic": self.args.joint_state_topic,
            "trajectory_topic": self.args.trajectory_topic,
            "terminal_time": self.terminal_time,
            "num_samples": len(self.rows),
            "stats": post_terminal_stats(self.rows, self.terminal_time),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        make_plot(
            [(self.label, self.rows, meta)],
            png_path,
            target_upj5=self.args.target_upj5,
            target_upj6=self.args.target_upj6,
        )

        self.get_logger().warn(f"Saved CSV: {csv_path}")
        self.get_logger().warn(f"Saved metadata: {meta_path}")
        self.get_logger().warn(f"Saved plot: {png_path}")


def run_record(args):
    rclpy.init()
    node = PassiveJointRecorder(args)
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


def run_compare(args):
    datasets = []
    for i, csv_path in enumerate(args.compare):
        rows = read_csv(csv_path)
        meta = read_meta(csv_path)
        label = meta.get("label") or os.path.splitext(os.path.basename(csv_path))[0]
        datasets.append((label, rows, meta))

    ensure_dir(args.output_dir)
    output_png = os.path.join(args.output_dir, f"{args.output_prefix}.png")
    make_plot(
        datasets,
        output_png,
        target_upj5=args.target_upj5,
        target_upj6=args.target_upj6,
    )

    summary_path = os.path.join(args.output_dir, f"{args.output_prefix}_summary.json")
    summary = {}
    for label, rows, meta in datasets:
        summary[label] = post_terminal_stats(rows, meta.get("terminal_time"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved comparison plot: {output_png}")
    print(f"Saved comparison summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-state-topic", default="/hanyang/joint_states")
    parser.add_argument("--trajectory-topic", default="/vp_sto_global_planner_node/actuated_reference")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--terminal-time", type=float, default=None)
    parser.add_argument("--start-without-trajectory", action="store_true")
    parser.add_argument("--label", default="run")
    parser.add_argument("--output-dir", default="/tmp/passive_joint_plots")
    parser.add_argument("--output-prefix", default="passive_joints")
    parser.add_argument("--target-upj5", type=float, default=None)
    parser.add_argument("--target-upj6", type=float, default=None)
    parser.add_argument("--compare", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.compare:
        run_compare(args)
    else:
        run_record(args)


if __name__ == "__main__":
    main()
