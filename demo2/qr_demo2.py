# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

from tensorforce.agents import PPOAgent,VPGAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import yaml
from qr_env2 import QR_ENV

with open('config.yml','r') as ymlfile:
    cfg=yaml.load(ymlfile)
env = QR_ENV(cfg,dset='train',is_train=True,verbose=True)


# Network as list of layers
network_spec = [
    dict(type='embedding',indices=374557,size=500),
	dict(type='expansion'),
    dict(type='conv2d',size=256,window=3,stride=2),
    dict(type='pool2d',window=2,stride=2),
    dict(type='conv2d',size=256,window=3),
    dict(type='pool2d',window=2,stride=2),
    dict(type='flatten'),
    dict(type='dense', size=256, activation='tanh'),
    dict(type='dense', size=256, activation='sigmoid')
]
'''agent = PPOAgent(
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec=network_spec,
    batch_size=4096,
    # Agent
    preprocessing=None,
    exploration=None,
    reward_preprocessing=None,
    # BatchAgent
    keep_last_timestep=True,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=0.99,
    # DistributionModel
    distributions_spec=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    summary_spec=None,
    distributed_spec=None
)'''

agent = VPGAgent(
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec= network_spec,
    batch_size=10
)

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=3000, max_episode_timesteps=2, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
