import json
from agent_teams.base_expert import BaseExpert


class ModelAdvisor(BaseExpert):
    ROLE_DESCRIPTION = """You are a senior operations research expert. Your role is to provide constructive, positive insights about the problem's domain terminology, key points, and essential nature—based strictly on the given text. Never critique, imply missing information, or speculate."""

    FORWARD_TASK = """Review the following problem description and your colleague's comment:

    {problem_description}

    {message_text}

    Provide 2–3 concise, helpful insights that may support accurate modeling. Each insight must belong to exactly one of these categories:
    - "Domain Terminology": Clarify what a term or symbol likely means in practice.
    - "Problem Key Point": Highlight an important detail in the wording that deserves attention (e.g., timing, scope, or specific conditions).
    - "Problem Essence": Characterize the fundamental nature of the problem using standard OR problem types such as job shop scheduling, assignment problem, or vehicle routing.

    All insights must be directly inferable from the provided text—do not invent assumptions.

    Output ONLY a JSON list in this exact format:
    [
      {{
        "category": "Domain Terminology" | "Problem Key Point" | "Problem Essence",
        "insight": "A clear, practical sentence."
      }}
    ]

    No other text, formatting, or explanation."""

    def __init__(self, model, temperature=0, base_url=None, api_key=None):
        super().__init__(
            name='Optimization Modeling Advisor',
            description='Proactive optimization expert that identifies potential modeling pitfalls and provides preventive guidance before formal problem formulation.',
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )

    def forward(self, problem, context):
        self.problem = problem
        output = super().forward(
            problem_description=problem,
            message_text=context,
        )

        try:
            parsed_output = json.loads(output)
            if isinstance(parsed_output, list):
                markdown_lines = []
                for item in parsed_output:
                    line = (
                        f"- **Category**: {item.get('category', 'N/A')}\n"
                        f"  **Insight**: {item.get('insight', 'N/A')}\n"
                    )
                    markdown_lines.append(line)
                answer = "\n\n".join(markdown_lines) + "\n"
            else:
                answer = output
        except (json.JSONDecodeError, TypeError):
            answer = output
        self.previous_answer = answer
        return answer
