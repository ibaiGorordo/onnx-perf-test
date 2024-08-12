import argparse
import time

import onnxruntime
import os
import sys

from .utils import tensors_to_string, create_test_inputs, provider_map
from .analyze_onnx_profile import analyze_onnx_profile


def create_session(model_path: str,
                   provider: str = "DEFAULT",
                   **kwargs) -> onnxruntime.InferenceSession:
    """ Create an InferenceSession with the specified provider """
    session_options = onnxruntime.SessionOptions()
    session_options.enable_profiling = True

    if provider.upper() in provider_map:
        provider = provider_map[provider.upper()]
    else:
        print(f"\033[91m\nError: Provider {provider} not found. Available options: {list(provider_map.keys())}\n\033[0m")
        sys.exit(1)
    return onnxruntime.InferenceSession(model_path, providers=provider, sess_options=session_options, **kwargs)


def run_test(model_path: str,
             provider: str = "DEFAULT",
             num_runs: int = 10,
             **kwargs) -> str:
    """ Run a performance test on the specified model """
    session = create_session(model_path, provider, **kwargs)
    print(f"Running performance test on {model_path} with provider {provider}")
    print(f"Model inputs: {tensors_to_string(session.get_inputs())}")
    print(f"Model outputs: {tensors_to_string(session.get_outputs())}")

    inputs = create_test_inputs(session)

    print("\nWarming up the session...")

    # Warm up the session
    session.run(None, inputs)

    print("Starting performance test...")
    for i in range(num_runs):
        start = time.perf_counter()
        session.run(None, inputs)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        print(f"Run {i + 1}: {elapsed_ms:0.2f} ms")

    prof_file = session.end_profiling()

    return prof_file


def get_parser():
    parser = argparse.ArgumentParser(description="Run a performance test on an ONNX model")
    parser.add_argument("onnx_model")
    parser.add_argument("--provider", type=str, help="Provider to use", default="DEFAULT")
    parser.add_argument("--num_runs", type=int, help="Number of runs", default=10)
    parser.add_argument("--output_dir", type=str, help="Output dir", default="")
    parser.add_argument("--draw", action="store_true", help="Draw results (Requires: matplotlib)")
    parser.add_argument("--keep_profiling_file", action="store_true", help="Keep the profiling file")
    return parser


def main():
    args = get_parser().parse_args()
    onnx_profile_file = run_test(args.onnx_model, args.provider, args.num_runs)

    analyze_onnx_profile(onnx_profile_file, args.output_dir, args.draw)

    if not args.keep_profiling_file:
        os.remove(onnx_profile_file)


if __name__ == '__main__':
    main()
