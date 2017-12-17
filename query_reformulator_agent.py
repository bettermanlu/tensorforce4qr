from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import Agent
import tensorflow as tf

from tensorforce import util, TensorForceError

class QueryReformulatorAgent(Agent):
    """
    `QueryReformulatorAgent`
    """
    def __init__(
        self,
        states_spec,
        actions_spec,
        batched_observe,
        summary_spec,
        qr_env
    ):
        """

        Args:
            states_spec: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions_spec: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            preprocessing: Optional list of preprocessors (e.g. `image_resize`, `grayscale`) to apply to state. Each
                preprocessor is a dict containing a type and optional necessary arguments.
            exploration: Optional dict specifying exploration type (epsilon greedy strategies or Gaussian noise)
                and arguments.
            reward_preprocessing: Optional dict specifying reward preprocessor using same syntax as state preprocessing.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
        """
        #ToDO: add more initilizations.
        self.summary_spec = summary_spec
      
        self.qr_env = qr_env
        
        self.episode = tf.Variable(
                    name='episode',
                    dtype=util.tf_dtype('int'),
                    trainable=False,
                    initial_value=0
                )
        self.timestep = tf.Variable(
                    name='timestep',
                    dtype=util.tf_dtype('int'),
                    trainable=False,
                    initial_value=0
                )                
                        
        super(QueryReformulatorAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=batched_observe)
        
        
        
    def act(self, states, deterministic=False):
        #super(QueryReformulatorAgent, self).act(states, deterministic) #?? do we need to call super's act()?
        #return a search action
        #
        print('in agent.act(), states = ')
        print(states)
        qi, qi_i, qi_lst, D_gt_id, D_gt_title = self.qr_env.get_samples(sample_num = 1)
        actions = qi_i, qi_lst, D_gt_id 
        return actions
        
    def observe(self, terminal, reward):
        #super(MemoryAgent, self).observe(terminal=terminal, reward=reward)
        #now feedback to the model to update its hyper-parameters
        print('agent.observe(), terminal = ')
        print(terminal)
        print('reward = ')
        print(reward)

    def reset(self):
        print('agent.reset(), do nothing.')
        
    def initialize_model(self):
        print('agent.initialize_model(), do nothing.') 
        
    def should_stop(self):
       #todo:
       return False
       
    def close(self):
       return                 
