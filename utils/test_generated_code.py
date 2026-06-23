import importlib
import json
import math
import os
import sys
import re
import subprocess


from utils.result import Result


class NullWriter:
    def write(self, s):
        pass

error_list = {}


def insert_status_check(code: str) -> str:
    """
    在 code 里找到 xxx.optimize() 后插入一段万能状态判断。
    输出统一关键字，方便 run_code 做正则匹配。
    """
    # 1. 匹配模型变量名
    m = re.search(r"(\w+)\.optimize\(\)", code)
    if not m:
        return code
    solver = m.group(1)

    # 2. 追加片段（与主代码共用 GRB 常量）
    addon = f'''
# --- auto-inserted status check ---
status = {solver}.status
if status == GRB.OPTIMAL:
    print("OPTIMAL", {solver}.ObjVal)
elif status == GRB.INFEASIBLE:
    print("INFEASIBLE")
elif status == GRB.UNBOUNDED:
    print("UNBOUNDED")
elif status == GRB.INF_OR_UNBD:
    print("INF_OR_UNBD")
elif status == GRB.TIME_LIMIT:
    print("TIME_LIMIT")
elif status == GRB.NODE_LIMIT:
    print("NODE_LIMIT")
elif status == GRB.ITERATION_LIMIT:
    print("ITERATION_LIMIT")
elif status == GRB.SOLUTION_LIMIT:
    print("SOLUTION_LIMIT")
elif status == GRB.USER_OBJ_LIMIT:
    print("USER_OBJ_LIMIT")
elif status == GRB.WORK_LIMIT:
    print("WORK_LIMIT")
elif status == GRB.INTERRUPTED:
    print("INTERRUPTED")
elif status == GRB.NUMERIC:
    print("NUMERIC")
else:
    print("OTHER")
# --- end ---
'''
    return code.rstrip() + '\n' + addon


def run_code(code):
    with open("temp/generated_code.py", "w", encoding="utf-8") as f:
        f.write(code)

    try:
        res = subprocess.run(
            ["python", "temp/generated_code.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="ignore",
        )
        out = res.stdout
        err = res.stderr
        # print("xxxxxxxxxxxxxxxxxxxxxxxx,", out)
        # with open("log.txt", "a", encoding="utf-8") as f:
        #     f.write(f'执行结果{out}')

        NUMBER = r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?'
        PATTERNS = [
            rf'Best objective\s+({NUMBER})',  # 最可靠
            rf'Optimal objective\s*[:=]?\s*({NUMBER})',
            rf'OPTIMAL(?:\s+[a-zA-Z:]+\s*)?({NUMBER})',
            rf'Objective value\s*[:=]\s*({NUMBER})',
            rf'Solution count \d+:\s+({NUMBER})',
        ]

        for pattern in PATTERNS:
            m = re.search(pattern, out)
            if m:
                return float(m.group(1)),None

        # 2. 已知无可行解
        no_solution_markers = [
            "INFEASIBLE",  # Gurobi: Model is infeasible
            "UNBOUNDED",  # Gurobi: Model is unbounded
            "INF_OR_UNBD",
            "Model is infeasible",
            "Infeasible model",
            "Unbounded model",
            "No optimal solution found",
            "No feasible solution",  # 常见表述
            "infeasible or unbounded",
            "could not find a feasible solution"
        ]
        if any(m in out for m in no_solution_markers):
            return "No Best Solution", None

        # 3. 收集错误信息
        error_info = {
            "stdout": out.strip(),
            "stderr": err.strip(),
            "status": "Runtime Error"
        }
        return "Runtime Error", error_info

    except Exception as e:
        error_info = {
            "exception": str(e),
            "status": "Execution Failed"
        }
        return "Execution Failed", error_info


# 修改后的测试函数
def test_generated_code(ground_truth, log_file=None, iteration="iter_1"):
    global error_list
    log_file = log_file or NullWriter()

    try:
        original_code = open("temp/generated_code.py", "r", encoding='utf-8').read()
        modified_code = insert_status_check(original_code)
        output,error_info= run_code(modified_code)
    except Exception as e:
        print(f"Execution failed: {e}")
        error_list[iteration] = {"exception": str(e), "status": "COMPILE_ERROR"}
        return Result.COMPILE_ERROR

    # 收集错误信息
    if error_info and (output == "Runtime Error" or output == "Execution Failed"):
        error_list[iteration] = error_info

    passed_num = 0
    total_num = 1
    # print('=' * 20)
    # print(f"测试输出为{output}")
    # print(type(ground_truth), ground_truth)
    # print()
    # log_file.write('Program Output:\n')
    # log_file.write(str(output) + '\n\n')
    # log_file.write('Ground Truth:\n')
    # log_file.write(str(ground_truth) + '\n')

    is_re = False
    if isinstance(output, tuple):
        output = output[0]

    if output is not None and ground_truth is not None:
        try:
            if type(output) in [int, float, complex]:
                is_passed = math.isclose(float(output), float(ground_truth), rel_tol=1e-3, abs_tol=1e-1)
            else:
                is_passed = output == ground_truth
        except BaseException as e:
            is_passed = False
    elif output is None and ground_truth is None:
        is_passed = True
    else:
        is_passed = False

    if output == "Runtime Error" or output == "Execution Failed":
        is_re = True

    if is_passed:
        passed_num += 1

    # log_file.write(f'Is passed: {is_passed}\n')
    # log_file.write('\n')
    # log_file.write('\n\n')
    # log_file.write(f'{passed_num}/{total_num} passed\n')
    is_correct = (passed_num == total_num)
    # log_file.write(f'is correct: {is_correct}\n')

    if is_re:
        return Result.COMPILE_ERROR
    if is_correct:
        return Result.ACCEPT
    else:
        return Result.WRONG_ANSWER


# 获取错误列表的函数
def get_error_list():
    return error_list
# 清空错误列表的函数
def clear_error_list():
    global error_list
    error_list = {}
def run_generated_code(code):
    modified_code = insert_status_check(code)
    # 执行代码
    output,error_list = run_code(modified_code)
    return output

def run_eval_code(code):
    modified_code = insert_status_check(code)
    # 执行代码
    output = run_code(modified_code)
    return modified_code,output

#
def test_origin_output(output, ground_truth, log_file=None):
    log_file = log_file or NullWriter()

    print('=' * 20)
    print(output)
    print(ground_truth)
    print()
    # log_file.write('=' * 15 + 'test result' + '=' * 15 + '\n')
    # log_file.write('Program Output:\n')
    # log_file.write(str(output) + '\n\n')
    # log_file.write('Ground Truth:\n')
    # log_file.write(str(ground_truth) + '\n')

    if isinstance(output, tuple):
        output = output[0]

    is_passed = False
    if output is not None and ground_truth is not None:
        try:
            if isinstance(output, (int, float, complex)):
                is_passed = math.isclose(float(output), float(ground_truth), rel_tol=1e-3, abs_tol=2e-1)
            else:
                is_passed = output == ground_truth
        except BaseException:
            is_passed = False
    elif output is None and ground_truth is None:
        is_passed = True

    # log_file.write(f'Is passed: {is_passed}\n')
    # log_file.write('\n')

    if is_passed:
        return Result.ACCEPT
    else:
        return Result.WRONG_ANSWER
#
# if __name__ == '__main__':
#
