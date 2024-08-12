import os
import argparse

import numpy as np
import pandas as pd
import json

from .utils import draw_results


def profile_to_df(profile_data):
    names = [node["name"] for node in profile_data]
    durations = [node["dur"] / 1000 for node in profile_data]
    op_names = [node["args"]["op_name"] if node["cat"] == "Node" else node["name"] for node in profile_data]

    # Get run ids
    run_start_ids = [0] + [i for i, name in enumerate(names) if name == "model_run"]
    run_ids = np.zeros(len(names), dtype=int)
    for j in range(len(run_start_ids) - 1):
        run_ids[run_start_ids[j] + 1:run_start_ids[j + 1] + 1] = j

    df = pd.DataFrame({"name": names, "run_id": run_ids, "op_name": op_names, "duration": durations})
    df.set_index(["name", "run_id", "op_name"], inplace=True)

    return df


def parse_profile(prof_file):
    with open(prof_file) as f:
        profile_data = json.load(f)

    df = profile_to_df(profile_data)

    # Skip Warmup (run_id=0)
    df = df[df.index.get_level_values("run_id") > 0]

    # Calculate stats per name but also keep the op_name
    stats = df.groupby(["name", "op_name"])["duration"].agg(["mean", "std"])
    stats["percent"] = stats["mean"] / stats["mean"]["model_run"].values[0] * 100
    stats = stats.sort_values("mean", ascending=False)

    # Calculate stats per op_name by first summing durations per run_id and then calculating stats
    op_stats = df.groupby(["op_name", "run_id"])["duration"].sum().unstack().agg(["mean", "std"], axis=1)
    op_stats["percent"] = op_stats["mean"] / op_stats["mean"]["model_run"] * 100
    op_stats = op_stats.sort_values("mean", ascending=False)

    return stats, op_stats


def print_full_stats(stats, op_stats):
    # For each operation, it will print first the operation stats, and after the full stats for that operation
    for op_name in op_stats.index:
        op_stat = op_stats.loc[op_name]
        print(f"\n{op_name}: {op_stat['mean']} ± {op_stat['std']} ms")
        node_stats = stats[stats.index.get_level_values("op_name") == op_name]
        for i in range(len(node_stats)):
            node_stat = node_stats.iloc[i]
            print(f"\t{node_stat.name[0]}: {node_stat['mean']} ± {node_stat['std']} ms")


def analyze_onnx_profile(onnx_profile_file, output_dir="", draw=False):
    stats, op_stats = parse_profile(onnx_profile_file)
    print_full_stats(stats, op_stats)

    if output_dir:
        file_name = os.path.basename(onnx_profile_file)
        stats.to_csv(f"{output_dir}/stats_{file_name}.csv")

    if draw:
        draw_results(stats)


def get_parser():
    parser = argparse.ArgumentParser(description="Analyze ONNX profile json")
    parser.add_argument("onnx_profile_file")
    parser.add_argument("--output_dir", type=str, help="Output dir", default="")
    parser.add_argument("--draw", action="store_true", help="Draw results (Requires: matplotlib)")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    analyze_onnx_profile(args.onnx_profile_file, args.output_dir, args.draw)


if __name__ == '__main__':
    main()
