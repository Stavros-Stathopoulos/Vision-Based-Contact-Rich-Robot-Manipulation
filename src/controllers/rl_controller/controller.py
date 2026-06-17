from src.controllers.base_controller import BaseController


class Controller(BaseController):
    def __init__(self):
        pass

    def reset(self, seed=None):
        pass

    def act(self, observation):
        action = None
        info = None
        return action, info
