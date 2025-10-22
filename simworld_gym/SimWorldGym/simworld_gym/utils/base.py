from simworld_gym.utils.unrealcv_basic import UnrealCV

class ActionBuffer:
    def __init__(self, max_size=5, unrealcv_client: UnrealCV=None):
        self.buffer = {}
        self.max_size = max_size
        self.unrealcv_client = unrealcv_client

    def push_action(self, action):
        if action.actor_name not in self.buffer:
            self.buffer[action.actor_name] = ActionQueue(self.max_size)
        self.buffer[action.actor_name].push_action(action)

    def insert_action(self, action):
        if action.actor_name not in self.buffer:
            self.buffer[action.actor_name] = ActionQueue(self.max_size)
        self.buffer[action.actor_name].insert_action(action)

    def pop_actions(self):
        actions = []
        actions_indexes = []
        messages = []
        for actor_name in self.buffer:
            action = self.buffer[actor_name].pop_action()
            if action is not None:
                actions.append(action.__str__())
                actions_indexes.append(action.action_index)
                messages.append(action.message)
        return actions, actions_indexes, messages

    def get_action_history(self, actor_name):
        if actor_name in self.buffer:
            return self.buffer[actor_name].__str__()
        else:
            return ""

    def send_actions(self):
        actions, actions_indexes, messages = self.pop_actions()
        if not actions:
            return [], [], []
            
        # Use list comprehension for filtering
        valid_indices = [i for i, idx in enumerate(actions_indexes) if idx not in (6, 100)]
        if not valid_indices:
            return actions, actions_indexes, messages
            
        filtered_actions = [actions[i] for i in valid_indices]
        self.unrealcv_client.client.request_batch(filtered_actions)
        
        return actions, actions_indexes, messages

    def display_actions(self):
        # write the actions in the buffer and update it into file
        with open("actions.txt", "w") as f:
            for actor_name in self.buffer:
                f.write(f"{actor_name}:\n")
                f.write(f"    {self.buffer[actor_name].__str__()}\n")

class ActionQueue:
    def __init__(self, max_size=5):

        self.queue = []
        self.max_size = max_size

    def push_action(self, action):
        if self.is_full():
            self.pop_action()
        self.queue.append(action)

    # insert action at the front of the queue
    def insert_action(self, action):
        self.queue.insert(0, action)

    def pop_action(self):
        if self.is_empty():
            return None
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.max_size

    def __str__(self):
        return "\n".join([action.__str__() for action in self.queue])

class Action:
    def __init__(self, actor_name, action_command, action_args: list[str], action_index: int, message: dict={}):
        self.actor_name = actor_name
        self.action_command = action_command
        self.action_args = action_args
        self.action_index = action_index
        self.message = message

    def __str__(self):
        if len(self.action_args) > 0:
            action_args = " ".join([str(arg) for arg in self.action_args])
        else:
            action_args = ""
        return f"vbp {self.actor_name} {self.action_command} {action_args}"
