import json
import os
import logging

import numpy as np

from agent_teams import (
    ModelAdvisor,
    ModelingExpert,
    Compiler,
    ParaExtractor,
    message_pool_p,
    message_pool_m,
    message_pool_c,
)

from utils.result import Result
from agent_teams.solver import Solver
from utils.message import Message
from utils.message_pool import MessagePool


def _init_logger(problem_name, log_dir):
    """Create a per-problem file logger."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{problem_name}.log")
    logger = logging.getLogger(f"agent_{problem_name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def causal_agent(
    ground_truth,
    problem,
    problem_name,
    model_name,
    ifRev,
    attempt,
    temperature,
    attention,
    path,
    custom_base_url=None,
    api_key=None,
    data_path=None,
    persist_dir=None,
):
    """Run Chain of Experts pipeline.

    Args:
        problem: problem description string.
        data_path: path to RAG knowledge file (optional, uses env default).
        persist_dir: path to Chroma persistence directory (optional).

    Returns:
        (answer, output, ref_acc)
    """
    logger = _init_logger(problem_name, os.path.join(path, "logs"))
    logger.info(f"Start processing problem: {problem_name}")

    # --- Build experts (pass RAG config down) ---
    experts_p = [
        ParaExtractor(model_name, temperature, custom_base_url, api_key),
        ModelAdvisor(model_name, temperature, custom_base_url, api_key),
    ]
    experts_m = [ModelingExpert(
        model_name, temperature, custom_base_url, api_key,
        data_path=data_path, persist_dir=persist_dir,
    )]
    experts_c = [Compiler(
        model_name, temperature, custom_base_url, api_key,
        data_path=data_path, persist_dir=persist_dir,
    )]
    all_experts = experts_p + experts_m + experts_c

    num_experts = len(all_experts)
    director = Solver(model_name, temperature, base_url=custom_base_url, api_key=api_key)
    message_pool = MessagePool(all_experts, visible_matrix=np.ones((num_experts, num_experts)))
    message_pools_m = message_pool_m.MessagePool(experts_m, visible_matrix=np.ones((len(experts_m), len(experts_m))), mode="local")
    message_pools_c = message_pool_c.MessagePool(experts_c, visible_matrix=np.ones((len(experts_c), len(experts_c))), mode="local")
    expert_stack = []

    def _log_pool(pool, stage=""):
        try:
            logger.info(f"[{stage}] Message Pool:\n{pool.get_current_message_text()}\n{'-'*60}")
        except Exception as e:
            logger.info(f"[{stage}] Failed to log message pool: {e}")

    # === Stage 1: Parameter Extraction ===
    for expert in experts_p:
        logger.info(f"\n[Param Stage] >>> Calling expert: {expert.name}")
        context_p = message_pool.get_closest_message_text()
        message_text = expert.forward(problem, context_p)
        logger.info(f"[Param Stage] <<< Output from {expert.name}:\n{message_text}\n{'='*60}")
        message_pool.add_message(Message(expert, message_text))
        expert_stack.append(expert)

    # === Stage 2: Mathematical Modeling ===
    for expert in experts_m:
        logger.info(f"\n[Model Stage] >>> Calling expert: {expert.name}")
        context_m = message_pool.get_current_message_text()
        message_text = expert.forward(problem, context_m, log_dir=os.path.join(path, "logs"))
        logger.info(f"[Model Stage] <<< Output from {expert.name}:\n{message_text}\n{'='*60}")
        message_pool.add_message(Message(expert, message_text))
        message_pools_m.add_message(Message(expert, message_text))
        expert_stack.append(expert)

    # === Stage 3: Code Generation ===
    for expert in experts_c:
        logger.info(f"\n[Code Stage] >>> Calling expert: {expert.name}")
        message_pool.set_current_viewer(expert.name)
        context_c = message_pool.get_closest_message_text()
        message_text = expert.forward(problem, context_c, log_dir=os.path.join(path, "logs"))
        logger.info(f"[Code Stage] <<< Output from {expert.name}:\n{message_text}\n{'='*60}")
        message_pool.add_message(Message(expert, message_text))
        message_pools_c.add_message(Message(expert, message_text))
        expert_stack.append(expert)

    # === Stage 4: Final Format ===
    from utils.test_generated_code import run_generated_code, test_generated_code, get_error_list, clear_error_list
    from utils.utils import extract_code_from_string

    original_answer = director.forward(problem, message_pool, attention)
    answer = original_answer
    code = extract_code_from_string(original_answer)

    os.makedirs('temp', exist_ok=True)
    with open('temp/generated_code.py', 'w', encoding='utf8') as f:
        f.write(code)
    clear_error_list()
    output = run_generated_code(code)
    result = test_generated_code(ground_truth, None)
    logger.info(f"[Initial Result] {result.name}")

    # === Reflection Loop ===
    ref_acc = 0
    if ifRev and result == Result.COMPILE_ERROR:
        current_error = get_error_list()
        for att in range(attempt):
            logger.info(f"\n[Reflection] --- Attempt {att + 1}/{attempt} ---")

            # Modeling expert revision
            for expert in experts_m:
                logger.info(f"\n[Ref-Model] >>> Calling expert: {expert.name}")
                try:
                    original_model = message_pools_m.messages[-1].message_text if message_pools_m.messages else ""
                except Exception:
                    original_model = ""
                latest_code_output = message_pools_c.messages[-1].message_text if message_pools_c.messages else ""
                latest_code = extract_code_from_string(latest_code_output)
                error_message_m = current_error
                model_tip = expert.get_memory_item()
                logger.info(f"[Ref-Model] Inputs:\n"
                            f"- Original Model:\n{original_model}\n"
                            f"- Error Message:\n{error_message_m}\n"
                            f"- Memory Tip:\n{model_tip}\n")
                _log_pool(message_pool, "Before Backward")
                message_text = expert.revision(problem, original_model, error_message_m, model_tip, latest_code)
                logger.info(f"[Ref-Model] <<< Output from {expert.name}:\n{message_text}\n{'='*60}")
                message_pool.add_message(Message(expert, message_text))
                message_pools_m.add_message(Message(expert, message_text))
                expert_stack.append(expert)

            # Compiler revision
            for expert in experts_c:
                logger.info(f"\n[Ref-Code] >>> Calling expert: {expert.name}")
                initial_code = extract_code_from_string(
                    message_pools_c.messages[-1].message_text if message_pools_c.messages else ""
                )
                error_message_c = current_error
                code_tip = expert.get_memory_item()
                logger.info(f"[Ref-Code] Inputs:\n"
                            f"- Initial Code:\n{initial_code}\n"
                            f"- Error Message:\n{error_message_c}\n"
                            f"- Memory Tip:\n{code_tip}\n")
                _log_pool(message_pool, "Before Backward")
                corrected_model = message_pools_m.messages[-1].message_text if message_pools_m.messages else ""
                message_text = expert.revision(problem, initial_code, code_tip, error_message_c, corrected_model)
                logger.info(f"[Ref-Code] <<< Output from {expert.name}:\n{message_text}\n{'='*60}")
                message_pool.add_message(Message(expert, message_text))
                message_pools_c.add_message(Message(expert, message_text))
                expert_stack.append(expert)

            # Re-generate
            original_answer = director.forward(problem, message_pool, attention)
            answer = original_answer
            code = extract_code_from_string(original_answer)
            with open('temp/generated_code.py', 'w', encoding='utf8') as f:
                f.write(code)
            clear_error_list()
            output = run_generated_code(code)
            result = test_generated_code(ground_truth, None)
            current_error = get_error_list()
            logger.info(f"[After Reflection] Result: {result.name}")
            if result == Result.ACCEPT:
                ref_acc += 1
                break

    logger.info(f"End processing problem: {problem_name} | Final result: {result.name}\n")
    return answer, output, ref_acc
