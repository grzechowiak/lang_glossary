
class ConfigAgents:
    """Configuration class"""
    def __init__(self):
            self.max_iterations = 2
            self.recursion_limit = 20
            self.llm_model = 'gpt-4o-mini' #"gpt-4o" #"gpt-4o-mini" #'gpt-4.1-nano'
            self.llm_temperature = 0.0