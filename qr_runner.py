import logging

from query_reformulator_env import QueryReformulatorEnv
from query_reformulator_agent import QueryReformulatorAgent
from tensorforce.execution import Runner

def main():
    max_episodes = 20
    max_timesteps = 20
    DATA_DIR = '/srv/local/work/sixilu2/sixilu2/github/queryreformulator/QueryReformulator'
    env = QueryReformulatorEnv(DATA_DIR=DATA_DIR,dset='train',is_train=True,verbose=True)

    agent = QueryReformulatorAgent(
      dict(shape=1, type='float'),
      dict(shape=1, type='float'),      
      batched_observe = 0,
      summary_spec=None,      
      qr_env = env
    )

    runner = Runner(agent, env)
    
    report_episodes = 10

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            logging.info("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            logging.info("Episode reward: {}".format(r.episode_rewards[-1]))
            #logging.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run(episodes=10,timesteps=10, max_episode_timesteps = 20,episode_finished=episode_finished)

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

if __name__ == '__main__':
    main()