"""Unified MessagePool with mode-based behavior for different expert teams."""
from utils.message import Message


class MessagePool:
    """Store and retrieve messages with optional visibility matrix.

    Modes:
      - 'global': get_closest_message_text returns ALL messages (used by main pipeline)
      - 'local':  get_closest_message_text returns only the last message (used by intra-team)
    """

    def __init__(self, experts, visible_matrix=None, mode="global"):
        self.messages = []
        self.experts = experts
        self.mode = mode
        self._current_viewer = None

        if visible_matrix is not None:
            self.visible_matrix = visible_matrix
            self.expert_name_to_id = {expert.name: i for i, expert in enumerate(experts)}
        else:
            self.visible_matrix = None
            self.expert_name_to_id = {}

    def set_current_viewer(self, expert_name):
        self._current_viewer = expert_name

    def add_message(self, message: Message):
        self.messages.append(message)

    def pop_message(self) -> Message:
        return self.messages.pop()

    def get_messages(self, expert_name):
        if self.visible_matrix is None or expert_name not in self.expert_name_to_id:
            return list(self.messages)
        id_ = self.expert_name_to_id[expert_name]
        visible_indices = self.visible_matrix[id_]
        message_list = []
        for message in self.messages:
            author_name = message.expert.name
            if author_name not in self.expert_name_to_id:
                continue
            target_id = self.expert_name_to_id[author_name]
            if visible_indices[target_id] == 1:
                message_list.append(message)
        return message_list

    def get_current_message_text(self):
        if not self.messages:
            return 'There is no message available, please ignore this section.\n'
        return ''.join(
            f"{m.expert.name}: ```{m.message_text}```\n"
            for m in self.messages
        )

    def get_closest_message_text(self):
        if not self.messages:
            return 'There is no message available, please ignore this section.\n'
        if self.mode == "local":
            # Return only the most recent message
            m = self.messages[-1]
            return f"{m.expert.name}: ```{m.message_text}```\n"
        # Global mode: return all messages
        return ''.join(
            f"{m.expert.name}: ```{m.message_text}```\n"
            for m in self.messages
        )

    def __len__(self):
        return len(self.messages)
