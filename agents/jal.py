import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

class JointActionLearner():

    """
    Implementation of Joint Action Learning

    Source: [FILL IN]
    """

    ### INITIALIZATION ###
    def __init__(
        self,
        agent_id: int,
        num_agents_per_team: int,
        teammate_idx: int,
        model,
        obs_size: int,
        act_size: int,
        learning_rate: float = 0.0005,
        batch_size: int = 64,
        gamma: float = 0.99,
        penalty: float = 0.0,
        max_memory: int  = 1000,
        last_n_reward: int = 100,
        num_joint_approx: int = 100
    ) -> None:

        # init team specifications 
        self.id = agent_id
        self.num_agents_per_team = num_agents_per_team
        self.teammate_idx = teammate_idx

        # init model
        self.model = model(obs_size, act_size**num_agents_per_team)
        self.target = model(obs_size, act_size**num_agents_per_team)
        self.optim = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.obs_size = obs_size
        self.act_size = act_size

        # init experienced replay
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.state_next_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.teammate_action_history = []

        # init model hyperparameters
        self.batch_size = batch_size # batch size
        self.gamma = gamma # discount factor
        self.penalty = penalty # penalty for finished episodes
        self.max_memory = max_memory # max memory of experience replay buffer
        self.num_joint_approx = num_joint_approx # number of experiences used to approximate joint action probabilities

        # init loss function
        self.loss_function = nn.HuberLoss()

        # init rewards
        self.episode_reward = 0
        self.running_reward = 0
        self.running_rewards = []
        self.last_n_reward = last_n_reward # number of rewards considered in running rewards
        self.episode_count = 0

    ### MODEL POLICY SAMPLING ###
    def sample_action_from_policy(self, state_t) -> torch.Tensor:

        """
        get action from model policy
        """

        with torch.no_grad():
            # get q values for each action
            q_vals = self.model(state_t)

        # reshape to get action vals for each agent
        q_vals = q_vals.view(self.act_size, self.act_size)

        # sample joint actions
        action_counts = self.sample_teammate_actions()

        # calculate action values = weighted q values
        action_vals = torch.sum(action_counts*q_vals, dim = 1) / self.num_joint_approx
        action_vals = action_vals.reshape(3, -1)

        # get action
        action = torch.argmax(action_vals, dim = 1)

        return action.numpy()

    ### MODEL UPDATIONS ###

    def update(self) -> None:

        if len(self.action_history) < self.batch_size:
            
            # replay buffer not large enough to train
            return

        state_sample, action_sample, reward_sample, state_next_sample, done_sample, teammate_action_sample = self.get_replay_batch()

        with torch.no_grad():

            Q_next_state = self.target(state_next_sample)
            Q_next_state = Q_next_state.view(-1, self.act_size, self.act_size)

            # sample teammate actions
            action_counts = self.sample_teammate_actions()

            # calculate action values = weighted q values
            next_state_action_vals = torch.sum(action_counts*Q_next_state, dim = 1) / self.num_joint_approx

            next_state_action_vals = next_state_action_vals.reshape(self.batch_size, 3, -1)

            # calculate estimated Q value by taking the maximum Q values of each action decision for the next state and summing across them
            Q_targets = reward_sample + self.gamma * torch.sum(torch.max(next_state_action_vals, dim = 2).values, dim = 1)

            # add penalty for finished episodes
            Q_targets = Q_targets * (1-done_sample) - self.penalty * done_sample
        
        action_sample[:, 1] += 3 # sideways actions have q indexes: 3-5 
        action_sample[:, 2] += 6 # rotation actions have q indexes: 6-8
        teammate_action_sample[:, 1] += 3
        teammate_action_sample[:, 2] += 6

        relevant_actions = torch.eye(self.act_size)[action_sample]
        relevant_actions = torch.max(relevant_actions, dim = 1).values
        relevant_actions = relevant_actions.reshape(self.batch_size, -1, 1)
        
        relevant_teammate_actions = torch.eye(self.act_size)[teammate_action_sample]
        relevant_teammate_actions = torch.max(relevant_teammate_actions, dim = 1).values
        relevant_teammate_actions = relevant_teammate_actions.reshape(self.batch_size, 1, -1)

        relevant_joint_actions = relevant_actions*relevant_teammate_actions

        self.model.zero_grad()
        q_values = self.model(state_sample)
        q_values = q_values.view(-1, self.act_size, self.act_size)

        Q_of_actions = torch.sum(torch.sum(q_values * relevant_joint_actions, dim = 2), dim = 1)

        loss = self.loss_function(Q_targets, Q_of_actions)
        loss.backward()
        self.optim.step()

    def update_target(self) -> None:

        """
        updating target model with new weights
        """

        self.target.load_state_dict(self.model.state_dict())

    def update_generation(self, best_agent):

        self.model.load_state_dict(best_agent.model.state_dict())
        self.target.load_state_dict(best_agent.model.state_dict())
        self.running_rewards = best_agent.running_rewards
        self.episode_reward_history = best_agent.episode_reward_history

    ### EXPERIENCE REPLAY ###
    def update_replay_history(self, state, action, next_state, reward, done) -> None:

        """
        updating replay history buffers
        """

        # change next state to current state if episode is finished (since next state for finished is empty)
        next_state = next_state if not done else state

        # only store actions of the current agent and its teammate
        action, teammate_action = action[self.id], action[self.teammate_idx]

        # update replay
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.state_next_history.append(next_state)
        self.done_history.append(done)
        self.teammate_action_history.append(teammate_action)

        # update episode reward
        self.episode_reward += reward

    def trim_replay_history(self) -> None:

        """
        trim buffer for efficiency
        """

        if len(self.reward_history) > self.max_memory:

            del self.reward_history[:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.action_history[:1]
            del self.done_history[:1]

    def get_replay_batch(self):

        """
        sample mini batches from experience replay buffer
        """
        
        # generate batch indexes
        batch = np.random.randint(
            low = 0,
            high = len(self.action_history),
            size = self.batch_size
        )

        # sample batches
        state_sample = torch.tensor(np.array([self.state_history[b] for b in batch]))
        action_sample = torch.tensor(np.array([self.action_history[b] for b in batch]))
        reward_sample = torch.tensor(np.array([self.reward_history[b] for b in batch]))
        state_next_sample = torch.tensor(np.array([self.state_next_history[b] for b in batch]))
        done_sample = torch.tensor(np.array([self.done_history[b] for b in batch]), dtype = torch.float32)
        teammate_action_sample = torch.tensor(np.array([self.teammate_action_history[b] for b in batch]))

        return state_sample, action_sample, reward_sample, state_next_sample, done_sample, teammate_action_sample

    def sample_teammate_actions(self):

        # replay buffer not large enough to train
        if len(self.action_history) < self.num_joint_approx:

            return torch.tensor(np.random.randint(0, 5, size = (self.act_size, self.act_size)))

        # generate batch indexes
        batch = np.random.randint(
            low = 0,
            high = len(self.action_history),
            size = self.num_joint_approx
        )

        # sample batches
        teammate_actions_sample = torch.tensor(np.array([self.teammate_action_history[b] for b in batch]))

        # get one hot encodinng of joint_actions
        teammate_actions_sample[:, 1] += 3
        teammate_actions_sample[:, 2] += 6

        # one hot encode actions
        relevant_actions = torch.eye(self.act_size)[teammate_actions_sample]
        relevant_actions = torch.max(relevant_actions, dim = 1).values

        # count actions across batches
        action_counts = torch.sum(relevant_actions, dim = 0)

        return action_counts

    ### REWARDS ###
    def update_episode_reward(self) -> None:

        """
        updates the history of episodes rewards
        """

        # adds reward of latest episode
        self.episode_reward_history.append(self.episode_reward)
        self.episode_count += 1

        # trim history
        if len(self.episode_reward_history) > self.last_n_reward:

            del self.episode_reward_history[:1]

        # calculate running reward
        self.running_reward = np.mean(self.episode_reward_history)

        # reset accumulator
        self.episode_reward = 0

    def update_running_rewards(self) -> None:

        """
        update running rewards history
        """

        self.running_rewards.append(self.running_reward)

    def get_running_reward(self) -> float:

        return self.running_reward

    ### MODEL SAVING ###
    def save_model(self, filename):

        torch.save(self.model.state_dict(), f'./weights/jal_{filename}.pt')
        print(f'weights saved at /weights/jal_{filename}.pt')

    ### PROGRESS TRACKING ###
    def print_training_progress(self, agent: int, timestep: int, generation: int, episode_time: float, total_time: float) -> None:

        """
        printing training information from the latest episode
        """

        template = "AGENT {}, EPISODE {}, FRAME {}, GENERATION {}: last reward = {:.2f}, running reward =  {:.2f}, episode time = {:.2f}, total time = {:.2f}"
        print(template.format(
            agent,
            self.episode_count,
            timestep,
            generation,
            self.episode_reward_history[-1],
            self.running_reward,
            episode_time,
            total_time
        ))

    def print_testing_progrees(self, agent: int, episode: int, episode_reward):

        template = "AGENT {}, EPISODE {}: reward {:.2f}"
        print(template.format(
            agent,
            episode,
            episode_reward
        ))

    def plot_running_rewards(self, update_steps: int) -> None:
        rr = np.array([update_steps] + self.running_rewards)
        np.savetxt("./plots/jal_runningrewards.csv", rr, delimiter=',')
        print('saved running rewards at /plots/jal_running_rewards.csv')

        plt.plot(np.arange(1, update_steps*len(self.running_rewards)+1, update_steps), self.running_rewards)
        plt.title('running rewards against frame count')
        plt.xlabel('frame count')
        plt.ylabel('running rewards')
        plt.show()
        plt.savefig('./plots/jal_training_plot.png')
        print('plot saved at /plots/jal_training_plot.png')