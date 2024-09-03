import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
import numpy as np
from typing import List, Tuple
from mlagents_envs.environment import ActionTuple

class SoccerTwosEnvWrapper():
    """
    SoccerTwos Environment Wrapper
    """

    def __init__(self, env) -> None :

        # init environment
        self.env = env
        self.env.reset()

        # init teams features
        self.teams = list(self.env.behavior_specs) # order: team1 and team0
        self.num_agents = 4
        self.num_inputs_per_agent = 8
        self.num_agent_per_team = self.num_agents // len(self.teams)
        self.actions = [None] * self.num_agents
        self.actions_set = [False] * self.num_agents

        # init environment specifications for each team
        self.specs = [self.env.behavior_specs[team] for team in self.teams]
        self.action_specs = [spec.action_spec for spec in self.specs]
        self.obssize = self.num_inputs_per_agent*sum(obs_spec.shape[0] for obs_spec in self.specs[0].observation_specs)
        self.actsize = 3*self.action_specs[0].discrete_size

    def reset(self) -> List[np.ndarray]:

        """
        resets the environment and updates decision, returns starting observation state
        """

        # reset environment
        self.env.reset()

        # update agent states (i.e deciding or terminal)
        self.update_steps()

        # get observation states for every agent
        states = [self.get_observation(agent) for agent in range(self.num_agents)]

        # reset actions
        self.reset_actions()

        return states

    def update_steps(self) -> None:

        """
        updates the states of each agent branch in the environment
        """

        # reset the states for each agent
        self.decision_steps = [None, None]
        self.terminal_steps = [None, None]

        # assign 
        for team_idx, team_behavior in enumerate(self.teams):

            self.decision_steps[team_idx], self.terminal_steps[team_idx] = self.env.get_steps(team_behavior)
           
    def step(self) -> Tuple[List[np.ndarray], List[np.ndarray], bool]:

        """
        take a step transition in environment
        returns next_obs, reward, done
        """

        # check if all agents have an action set
        if self.actions_set != [True]*4:
            raise Exception('Train Error: Not all actions are set for every agent')

        # take step
        self.env.step()

        # update environment variables
        self.update_steps()

        # get next observation states
        next_obs = [self.get_observation(agent) for agent in range(self.num_agents)]

        # get rewards
        rewards = [self.get_reward(agent) for agent in range(self.num_agents)]

        # get done state of agents in each team
        done1 = list(self.decision_steps[1]) == []
        done0 = list(self.decision_steps[0]) == []


        # print('done', done0, list(self.decision_steps[0]), list(self.env.get_steps("SoccerTwos?team=1")[0]))
        assert done1 == done0

        # reset actions
        self.reset_actions()

        return (next_obs, rewards, [done0]*self.num_agents)
        
    def set_action(self, agent: int, action: np.ndarray) -> None:

        """
        sets the policy of current step for a particular team (2 agents)
        """
        
        # check if action exists
        if self.actions_set[agent]:

            raise Exception(f"Train Error: Action already set for agent {agent}")
        
        # multiplies the action tensor across all action branches
        action = np.tile(action, (self.num_inputs_per_agent, 1))
    
        self.actions[agent] = action
        self.actions_set[agent] = True

        # if all actions have been set, then set action in environment at once
        if self.actions_set == [True]*self.num_agents:

            self.set_env_action()
    
    def set_env_action(self) -> None:

        """
        prepare actions in environment for next step
        """

        for team_idx, team in enumerate(self.teams):

            action = np.concatenate(
                (self.actions[team_idx*2], self.actions[team_idx*2+1]),
                axis = 0
            )

            # format actions into ActionTuple
            action = self.format_action(team_idx, action)

            self.env.set_actions(team, action)

    def format_action(self, team_idx: int, action: np.ndarray) -> ActionTuple:

        """
        formats action tensor into mlagents ActionTuple
        """
        action_spec = self.action_specs[team_idx]

        # get number of inputs required for action
        decision_steps = self.decision_steps[team_idx]

        # create an blank slate action to fill
        a = action_spec.empty_action(len(decision_steps))

        # fill action with policy
        a.add_discrete(action)

        return a

    def reset_actions(self) -> None:

        self.actions = [None]*self.num_agents
        self.actions_set = [False]*self.num_agents

    def get_observation(self, agent: int) -> np.ndarray:

        """
        get the current observation state of an agent
        """

        # get agent states
        team_idx = agent // len(self.teams)
        decision_steps = self.decision_steps[team_idx]
        
        # get observation array for team
        team_observations = decision_steps.obs
        forward_obs, backward_obs = team_observations

        # concatenate forward and backward observations
        all_obs = np.concatenate((forward_obs, backward_obs), axis = 1)

        # get agent-specific observation
        observation = all_obs[:self.num_inputs_per_agent, :] if agent%2 else all_obs[self.num_inputs_per_agent:, :]

        # reshape observation into a one dim tensor
        observation = observation.reshape(-1,)
        return observation
    
    def get_reward(self, agent: int) -> float:

        """
        get the reward of the agent (sum of all branches)
        """

        # get agent states
        team_idx = agent // len(self.teams)
        player_idx = agent % len(self.teams)
        terminal_steps = self.terminal_steps[team_idx]

        # get rewards for entire team
        team_rewards = terminal_steps.group_reward

        # get agent specific rewards
        reward = team_rewards[player_idx] if len(team_rewards) > 0 else np.float32(0)

        return reward

    def get_random_action(self, agent: int) -> np.ndarray:

        """
        sample a random action for the given agent
        """
        # get relevant information
        action_spec = self.action_specs[agent//len(self.teams)]
        team = agent // len(self.teams)
        decision_steps = self.decision_steps[team]

        # generate random action
        random_action = action_spec.random_action(1)

        # get discrete part of ActionTuple
        random_action = random_action._discrete

        return random_action.reshape(-1,)

    def close(self):

        """
        close environment
        """

        self.env.close()