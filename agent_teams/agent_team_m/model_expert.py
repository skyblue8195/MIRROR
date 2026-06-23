from agent_teams.base_expert import BaseExpert
from utils.rag import KnowledgeRetriever


class ModelingExpert(BaseExpert):
    ROLE_DESCRIPTION = 'You are a modeling assistant specialized in the field of Operations Research for mathematical formulation.'

    FORWARD_TASK = '''Your task is to formulate a precise mathematical optimization model based on the problem description below.

    Relevant modeling examples from knowledge base (focus on their formulation structure and methodology only):
    {knowledge_context}

    You can refer to the parameters and other information provided by your colleagues.:
    {message_text}

    Now, formulate a model for this problem:
    {problem_description}

    Formulate the model strictly in the following JSON format:
    {{
      "VARIABLES": "A concise description about variables and its shape or type",
      "CONSTRAINTS": "A mathematical formula expressing all constraints",
      "OBJECTIVE": "A mathematical formula for the objective function"
    }}

    Important:
    1. Learn formulation patterns from examples without replicating their specific details
    2. Actively incorporating colleagues' advice in your formulation
    3. Base your formulation primarily on the original problem description
    4. Output ONLY the JSON object without any additional text
    '''

    REVISION_TASK = '''You are a senior operations research expert performing debugging and knowledge distillation.
           Your task is to Diagnose the root cause and produce a fully corrected model.
           and Extract a structured tip encoding: what went wrong, where, and how to fix it — for future reuse. based on:
           - The original problem description
           - The incorrect model
           - The solver error
           - The last tip

           ### Input
           Problem Description:
           {problem_description}

           ###Original (Incorrect) Model:
           {original_model}

           ###Execution Error (from the code):
           {error_message}

           ### The Last tip for reference
           The last tip is provided for reference only and may be incorrect — please carefully verify and use your judgment
           {last_tip}.

           ### Output Requirement
           Return a tuple of two JSON objects:
           First: A tip JSON with EXACTLY the following keys::
           {{
             "tip_type":modeling:,
             "scenario": "The real-world application domain of the problem (e.g., energy, aviation, logistics, manufacturing, supply chain, transportation, etc.)",
             "error_statement": "The most relevant sentence or phrase from the problem description that relates to the modeling error (quote verbatim93
             "correct_component": "The precise corrected formulation (e.g., a constraint formula), not the full model",
             "incorrect_model": The incorrect formulation (from the original_model,e.g., a constraint formula), not the full model
           }}
           Second: The fully corrected model in standard JSON format:
           {{
            "VARIABLES": "A concise description about variables and its shape or type ,
            "CONSTRAINTS": "Corrected constraint formulations ",
            "OBJECTIVE": "Adjusted objective if necessary (with rationale)"
           }}
           Return exactly:
           TIP_JSON
           <split>
           CORRECTED_MODEL_JSON
           Output ONLY the tuple of two JSONs — no prefix, no suffix.
       '''

    def __init__(self, model, temperature=0, base_url=None, api_key=None,
                 data_path=None, persist_dir=None):
        super().__init__(
            name='Modeling Expert',
            description='Proficient in constructing mathematical optimization models based on the extracted information.',
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
        self.correction_history = []
        self.retriever = KnowledgeRetriever(
            mode="modeling", data_path=data_path, persist_dir=persist_dir, api_key=api_key,
        )

    def forward(self, problem, context, log_dir=None):
        message_text = context
        problem_description = problem
        knowledge_context = self.retriever.retrieve(problem_description)
        if log_dir is not None:
            import os
            log_file = os.path.join(log_dir, "rag_examples.log")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"\n\n[Problem]\n{problem_description}\n\n[Retrieved Modeling Examples]\n{knowledge_context}\n{'-' * 80}")

        return super().forward(
            problem_description=problem_description,
            message_text=message_text,
            knowledge_context=knowledge_context,
        )

    def revision(self, problem, original_model, error_message, last_model_tip, latest_code):
        output_text = super().backward(
            problem_description=problem,
            original_model=original_model,
            error_message=error_message,
            last_tip=last_model_tip,
            last_code=latest_code,
        )

        from utils.json_parser import parse_json_tuple_with_split
        default_tip = (
            '{"tip_type": "modeling", "scenario": "unknown", '
            '"error_statement": "Failed to parse structured output from LLM", '
            '"correct_component": "", "incorrect_model": ""}'
        )
        blocks = parse_json_tuple_with_split(output_text, expected_count=2, default_first=default_tip)

        tip_raw, model_raw = blocks[0], blocks[1]
        self.correction_history.append(tip_raw)
        return model_raw

    def get_memory_item(self):
        if not self.correction_history:
            return "No tip for reference"
        return self.correction_history[-1]
