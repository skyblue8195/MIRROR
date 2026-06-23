from agent_teams.base_expert import BaseExpert

class Solver(BaseExpert):
    ROLE_DESCRIPTION = 'You are a director that responsible for giving the final answer'
    FORWARD_TASK = '''Your colleague programmer has given his answer:
{message_text}
This answer has not been formatted. You need to format the code .
You also need to return the optimized variables.
Important: Your final code should strictly use same name exactly.
{attention}
Don't forget to import the library. Don't give any example usage.
'''

    def __init__(self, model, temperature=0, base_url=None, api_key=None):
        super().__init__(
            name='Solver',
            description='Final code formatter ',
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )

    def forward(self, problem, workspace, attention):
        message_text = workspace.get_closest_message_text()
        answer = super().forward(
            message_text=message_text,
            attention=attention
        )
        self.previous_code = answer
        self.attention = attention
        return answer
