"""
MIRROR - Run RAG Evaluation Experiment.

Runs the Chain-of-Experts causal-agent pipeline over a JSONL dataset,
evaluates each problem, and reports aggregate accuracy / error metrics.

NOTE:
    This script runs the standard Chain-of-Experts pipeline (parameter
    extraction -> modeling -> code generation -> reflection).  It does NOT
    include RAG retrieval - the --data_path and --persist_dir arguments are
    reserved for future integration but have no effect in the current version.

Usage:
    python run_rag_exp.py --dataset datasets/ComplexOR.jsonl --model qwen-plus-2025-09-11
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from main import causal_agent
from utils.result import Result
from utils.test_generated_code import test_generated_code
from utils.utils import extract_code_from_string


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run MIRROR Chain-of-Experts evaluation over a JSONL dataset."
    )
    parser.add_argument(
        "--dataset", type=str, default="datasets/ComplexOR.jsonl",
        help="Path to the evaluation dataset (JSONL format).",
    )
    parser.add_argument(
        "--dataname", type=str, default="",
        help="Dataset identifier. Auto-extracted from filename if empty.",
    )
    parser.add_argument(
        "--log_dir", type=str, default="../log",
        help="Directory to store experiment logs and outputs.",
    )
    parser.add_argument(
        "--model", type=str, default="qwen-plus-2025-09-11",
        help="LLM model name.",
    )
    parser.add_argument(
        "--attempts", type=int, default=3,
        help="Maximum number of reflection/revision attempts on compile errors.",
    )
    parser.add_argument(
        "--enable_revision", action="store_true", default=True,
        help="Enable modeling/code revision on compile errors.",
    )
    parser.add_argument(
        "--enable_few_shot", action="store_true", default=True,
        help="Enable few-shot prompting.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--attention", type=str,
        default=(
            "The code must not contain any function definition (i.e., no def():); "
            "it must directly return the objective value."
        ),
        help="Additional instruction appended to the prompt.",
    )
    parser.add_argument(
        "--base_url", type=str, default="",
        help="Custom API base URL.",
    )
    parser.add_argument(
        "--api_key", type=str, default="",
        help="API key for the LLM provider.",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="[Reserved] Path to RAG knowledge .md file. Not active in this version.",
    )
    parser.add_argument(
        "--persist_dir", type=str, default=None,
        help="[Reserved] Path to ChromaDB persistence directory. Not active in this version.",
    )
    return parser.parse_args()


def _extract_dataname(dataset_path):
    filename = os.path.basename(dataset_path)
    stem = filename.rsplit(".", 1)[0]
    match = re.match(r"^([a-zA-Z]+)", stem)
    return match.group(1) if match else "Unknown"


def _run_single_problem(question, problem_name, args, output_dir):
    problem = question["prompt"]
    ground_truth = question["en_answer"]

    answer, output, ref_acc = causal_agent(
        ground_truth=ground_truth,
        problem=problem,
        problem_name=problem_name,
        model_name=args.model,
        ifRev=args.enable_revision,
        attempt=args.attempts,
        temperature=args.temperature,
        attention=args.attention,
        path=output_dir,
        custom_base_url=args.base_url,
        api_key=args.api_key,
        data_path=args.data_path,
        persist_dir=args.persist_dir,
    )

    with open(os.path.join(output_dir, f"{problem_name}_answer.txt"), "w", encoding="utf-8") as f:
        f.write(answer)

    code = extract_code_from_string(answer)
    with open(os.path.join(output_dir, f"{problem_name}_generated_code.py"), "w", encoding="utf-8") as f:
        f.write(code)

    result = test_generated_code(ground_truth, None)

    return result, output


def run_experiment(args):
    dataname = args.dataname or _extract_dataname(args.dataset)
    timestamp = str(round(time.time()))
    log_dir = os.path.join(args.log_dir, f"{dataname}_{timestamp}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path("temp").mkdir(exist_ok=True)

    print(f"Dataset : {dataname}")
    print(f"Model   : {args.model}")
    print(f"Revision: {args.enable_revision} (max {args.attempts} attempts)")
    print(f"Log dir : {log_dir}")

    with open(args.dataset, "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f if line.strip()]

    total = len(data_list)
    correct = 0
    wrong_answer = 0
    compile_error = 0
    reflection_success = 0

    pbar = tqdm(total=total, desc="Evaluating")

    for idx, question in enumerate(data_list):
        result, output = _run_single_problem(question, str(idx), args, log_dir)
        time.sleep(1)

        if result == Result.ACCEPT:
            correct += 1
        elif result == Result.WRONG_ANSWER:
            wrong_answer += 1
        elif result == Result.COMPILE_ERROR:
            compile_error += 1

        pbar.update()
        pbar.set_description(
            f"Acc:{correct / (idx + 1) * 100:.1f}% "
            f"WA:{wrong_answer / (idx + 1) * 100:.1f}% "
            f"CE:{compile_error / (idx + 1) * 100:.1f}%"
        )
        print()

    pbar.close()

    print("\n" + "=" * 50)
    print(f"Experiment Summary - {dataname}")
    print(f"  Total problems : {total}")
    print(f"  Accuracy       : {correct / total * 100:.2f}%")
    print(f"  Wrong answer   : {wrong_answer / total * 100:.2f}%")
    print(f"  Compile error  : {compile_error / total * 100:.2f}%")
    print(f"  Reflection SR  : {reflection_success / total * 100:.2f}%")
    print("=" * 50)


def main():
    args = _parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
