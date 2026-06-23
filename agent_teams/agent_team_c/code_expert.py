from agent_teams.base_expert import BaseExpert
from utils.utils import extract_code_from_string
from utils.rag import KnowledgeRetriever
from utils.json_parser import parse_json_tuple_with_split
import json


class Compiler(BaseExpert):
    ROLE_DESCRIPTION = 'You are a Python programmer specializing in operations research and optimization.'
    FORWARD_TASK = '''You are tasked with developing an efficient Python program to solve the following problem:
    {problem_description}

    Reference modeling approach from colleague (use as guidance for formulation structure):
    {message_text}

    Relevant code examples (focus on implementation patterns and structure, not specific variables or constraints):
    {knowledge_context}

    Requirements:
    1. Use the Gurobipy library for implementation
    2. Do not include code usage examples
    3. Create a solution based primarily on the problem description
    4. Learn implementation patterns from examples without replicating specific details
    5. Integrate colleague's modeling approach into your implementation
    6. Output only the code without any function definitions
    '''
    REVISION_TASK = '''You are debugging an optimization implementation that encountered errors during execution.
           The original problem and attempted solution are as follows:

           ### Problem Context
           {problem_description}

           ### Original Code
           {initial_code}

           ### Execution Error
           {error_message}
           ### The Last tip for reference
           - The last tip is provided for reference only and may be incorrect — please carefully verify and use your judgment
            {last_tip}.
            Your goal is twofold:
            1. Diagnose the root cause and produce a fully corrected implementation.
            2. Extract a structured tip encoding: what went wrong, where, and how to fix it — for future reuse.

            Output a tuple of two JSON objects:
            (
          {{
             "tip_type": code:,
            "scenario": "The real-world application domain of the problem (e.g., energy, aviation, logistics, manufacturing, supply chain, transportation, etc.)",
            "error_statement": "The exact sentence or phrase from the problem description that relates to the code error (quote verbatim)",
            "code_error_location": "some of: 'variable_definition', 'constraint_addition', 'objective_setting', 'solver_invocation', 'data_processing',if there are many code errors,return a json structure",
            "correct_code_snippet": "Only the corrected code lines (e.g., fixed constraint addition), not the full code",
            "incorrect_code_snippet": "The original erroneous code lines as a string,not the full code"
          }},
          "Full corrected Python code using Gurobipy"
            )
            Rules:
        - Do NOT add any text before or after the tuple.
        - Keep `correct_code_snippet` minimal and targeted to the error.
        - The second element must be the complete, executable corrected code.
        - Base `error_statement` strictly on the problem description, not the code comments.
          Return exactly:
            TIP_JSON
            <split>
            CORRECTED_CODE_JSON
    '''

    def __init__(self, model, temperature=0, base_url=None, api_key=None,
                 data_path=None, persist_dir=None):
        super().__init__(
            name='Programming Expert',
            description='Skilled in programming and coding, capable of implementing the optimization solution in a programming language.',
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
        self.debug_history = []
        self.retriever = KnowledgeRetriever(
            mode="code", data_path=data_path, persist_dir=persist_dir, api_key=api_key,
        )

    def forward(self, problem, context, log_dir=None):
        message_text = context
        knowledge_context = self.retriever.retrieve(problem)
        if log_dir is not None:
            import os
            log_file = os.path.join(log_dir, "rag_examples.log")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"\n\n[Problem]\n{problem}\n\n[Retrieved Code Examples]\n{knowledge_context}\n{'-' * 80}")

        output = super().forward(
            problem_description=problem,
            message_text=message_text,
            knowledge_context=knowledge_context,
        )
        return extract_code_from_string(output)

    def revision(self, problem, initial_code, error_message, last_code_tip, corrected_model):
        output_text = super().backward(
            problem_description=problem,
            initial_code=initial_code,
            error_message=error_message,
            last_tip=last_code_tip,
            message_text=corrected_model,
        )

        default_tip = json.dumps({
            "tip_type": "code",
            "scenario": "unknown",
            "error_statement": "Failed to parse structured output from LLM",
            "code_error_location": "",
            "correct_code_snippet": "",
            "incorrect_code_snippet": "",
        }, ensure_ascii=False)

        blocks = parse_json_tuple_with_split(output_text, expected_count=2, default_first=default_tip)
        tip_raw, corrected_code = blocks[0], blocks[1]

        self.debug_history.append(tip_raw)
        return corrected_code

    def get_memory_item(self):
        if not self.debug_history:
            return "No tip for reference"
        return self.debug_history[-1]
