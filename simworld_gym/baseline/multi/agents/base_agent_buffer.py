class BaseAgentBuffer(object):
    """ Base Agent Buffer """
    def __init__(self, action_space=None, action_config=None, action_buffer=None, agent_controller=None):
        self.action_space = action_space
        self.action_config = action_config
        self.action_buffer = action_buffer
        self.agent_controller = agent_controller
        self.availability = False
        self.is_finished = False
    
    def act(self, obs):
        pass

    def send_action_to_buffer(self, action):
        self.action_buffer.push_action(action)

    def update_agent_availability(self):
        self.availability = self.agent_controller.return_agent_availability()
