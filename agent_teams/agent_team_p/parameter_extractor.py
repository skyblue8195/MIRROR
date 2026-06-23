from agent_teams.base_expert import BaseExpert


class ParaExtractor(BaseExpert):
    ROLE_DESCRIPTION = 'You are an assistant that extracts parameters and their types or shape from the given problem.'
    FORWARD_TASK = '''
    Please Extract parameters along with their concise definitions from the problem description:
    {problem_description}
    The comment from your colleague is:
    {message_text}
     **Key Principles for Parameter Type Inference:**
    1. **Analyze real-world meaning**: Determine if the parameter represents countable discrete units (int) or measurable continuous quantities (float)
    2. **Consider implementation feasibility**: Can this parameter logically take fractional values in practical use?
    3. **Verify operational suitability**: Ensure the chosen type supports all required mathematical operations
        without compromising the model's correctness or executability.
     Your output should be in JSON format as follows:
    {{
        "Parameter1": {{"Type": "The parameter's data type or shape", "Definition": "A brief definition of the parameter"}},
        "Parameter2": {{"Type": "The parameter's data type or shape", "Definition": "A brief definition of the parameter"}},
        ...
    }}
    Provide only the requested JSON output without any additional information.
    '''

    def __init__(self, model, temperature=0, base_url=None, api_key=None):
        super().__init__(
            name='Parameter Extractor',
            description='Expert at analyzing optimization problem descriptions to extract key parameters, their data types or shapes, '
                        'and precise semantic definitions for mathematical modeling.',
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )

    def forward(self, problem, context):
        self.problem = problem
        return super().forward(
            problem_description=problem,
            message_text=context,
        )
