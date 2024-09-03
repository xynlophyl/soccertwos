import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
from agents.random_agent import randomAgent
import argparse
import os

# root = r'C:/Users/nlinl/Desktop/Important/EECS6892/project/ml-agents/training-envs-executables'

def main(root, agent):

    # set up environment
    env = UE(file_name = root + '/' +'SoccerTwos.exe')

    # loading agent
    agent_dict = {
        'random': randomAgent
    }
    agent = agent_dict[agent]

    # start up environment
    try:

        env.reset()

        # behavior / policy that the given agent will be acting out
        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]
        action_spec = spec.action_spec

        for episode in range(5):

            episode_rewards = run_episode(env, spec, action_spec, behavior_name)
            print(f'total rewards for episode {episode} is {episode_rewards}')
    
    except KeyboardInterrupt:

        env.close()
        print('KeyboardInterrupt: Environment closed successfully')


def run_episode(env, spec, action_spec, behavior_name):

    # reset environment
    env.reset()

    # check agent status
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # setting up learning parameters
    agent = randomAgent()
    tracked_agent = -1
    done = False
    episode_rewards = 0
    tracked_agent = decision_steps.agent_id[0] # the agent we are training

    # generating random action for each agent

    while not done:

        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]

        # get observation tensor
        observation_spec = decision_steps.obs

        # generate action
        action = agent.get_action(observation_spec, action_spec)
        env.set_actions(behavior_name, action)
        
        # take a step in the environment
        env.step()

        # resample environment agents
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if tracked_agent in decision_steps:
            episode_rewards += decision_steps[tracked_agent].reward
        
        if tracked_agent in terminal_steps:
            episode_rewards += terminal_steps[tracked_agent].reward
            done = True
        
    return episode_rewards

if __name__ == '__main__':

    agent_list = ['random', 'dqn']

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', required=True)
    parser.add_argument('--agent', '-a', required=True, choices = agent_list)
    parser.add_argument('--epochs', '-e', default=20)

    args = parser.parse_args()

    if not os.path.exists(args.path + '/' +'SoccerTwos'):
        raise NotADirectoryError('(-p) SoccerTwos executable does not exist in specificed path')
    
    main(
        args.path, 
        args.agent
    )
