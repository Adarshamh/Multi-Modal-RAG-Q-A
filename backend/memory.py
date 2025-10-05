import uuid
from collections import defaultdict

class Memory:
    def __init__(self):
        self.sessions = defaultdict(list)

    def add_message(self, session_id, role, content):
        self.sessions[session_id].append({"role": role, "content": content})

    def get_session(self, session_id):
        return self.sessions[session_id]

memory = Memory()
