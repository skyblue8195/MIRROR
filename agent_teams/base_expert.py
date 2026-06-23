import os
from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatTongyi


class BaseExpert:
    """Expert that wraps a DashScope chat model with direct message invocation."""

    def __init__(
        self, name, description, model, temperature=0,
        base_url=None, api_key=None, max_tokens=None,
    ):
        self.name = name
        self.description = description
        self.model = model

        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        base_url = base_url or os.getenv("DASHSCOPE_BASE_URL")

        self.llm = ChatTongyi(
            model_name=model,
            temperature=temperature,
            dashscope_api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            max_retries=0,
        )

        self._role_description = self.ROLE_DESCRIPTION
        self._forward_task = self.FORWARD_TASK

        if hasattr(self, 'REVISION_TASK'):
            self._revision_task = self.REVISION_TASK

    def _build_messages(self, task_template: str, **kwargs) -> list:
        """Build system + user messages from template strings."""
        system_text = self._role_description
        user_text = task_template.format(**kwargs)
        return [
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ]

    def forward(self, **kwargs):
        messages = self._build_messages(self._forward_task, **kwargs)
        return self.llm.invoke(messages).content

    def backward(self, **kwargs):
        if hasattr(self, '_revision_task'):
            messages = self._build_messages(self._revision_task, **kwargs)
            return self.llm.invoke(messages).content
        raise NotImplementedError("revision method not implemented for this expert.")

    def __str__(self):
        return f'{self.name}: {self.description}'
