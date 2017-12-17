from tensorforce.models import Model

class QueryReformulatorModel(Model):
    def __init__(
        self,
        action_values #TODO: add qr related parameters.
    ):
        self.action_values = action_values
    
    def reset(self):
      pass
      
      
        
    