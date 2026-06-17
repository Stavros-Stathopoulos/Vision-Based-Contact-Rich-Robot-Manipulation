from src.controllers.base_controller import BaseController


class Controller(BaseController):
    def __init__(self, action_spec):
        self.action_spec = action_spec

    def reset(self, _seed=None):
        pass

    def act(self, _observation):
        action = self.action_spec[0] + (self.action_spec[1] - self.action_spec[0]) * 0.5
        return action, {}
