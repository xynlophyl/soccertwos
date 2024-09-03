import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

class DeepQN():

    """
    Implementation of Deep Q-Learning algorithm adapted from Deepmind's Deep Q Learning in Atari
    
    Source: https://deepmind.com/research/open-source/dqn
    """

    def __init__(
        self,
        model,
        in_size,
        out_size,
        learning_rate: float = 0.0005,
        batch_size: int = 64,
        gamma: float = 0.99,
        penalty: float = 0.0,
        max_memory: int  = 1000,
        last_n_reward: int = 100
    ) -> None:

        # init model
        self.model = model(in_size, out_size)
        self.target = model(in_size, out_size)
        self.optim = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.in_size = in_size
        self.out_size = out_size
    

        # init experienced replay
        self.state_history = []
        self.action_history = []
        self.rewards_history = []
        self.state_next_history = []
        self.done_history = []
        self.episode_reward_history = []

        # init model hyperparameters
        self.batch_size = batch_size # batch size
        self.gamma = gamma # discount factor
        self.penalty = penalty # penalty for finished episodes
        self.max_memory = max_memory

        # init loss function
        self.loss_function = nn.HuberLoss()

        # init rewards
        self.episode_reward = 0
        self.running_reward = 0
        self.running_rewards = []
        self.last_n_reward = last_n_reward # number of rewards considered in running rewards
        self.episode_count = 0

    def sample_action_from_policy(self, state_t) -> torch.Tensor:

        """
        get action from model policy
        """

        with torch.no_grad():
            action_vals = self.model(state_t)

        action_vals = action_vals.view(3,-1)
        action = torch.argmax(action_vals, dim = 1)

        return action.numpy()

    def update_replay_history(self, state, action, next_state, reward, done) -> None:

        """
        updating replay history buffers
        """

        # change next state to current state if episode is finished (since next state for finished is empty)
        next_state = next_state if not done else state 

        # update replay
        self.state_history.append(state)
        self.action_history.append(action)
        self.rewards_history.append(reward)
        self.state_next_history.append(next_state)
        self.done_history.append(done)

        # update episode reward
        self.episode_reward += reward

    def trim_buffer(self) -> None:

        """
        trim buffer for efficiency
        """

        if len(self.rewards_history) > self.max_memory:

            del self.rewards_history[:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.action_history[:1]
            del self.done_history[:1]

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

    def update(self) -> None:

        """
        fitting model using dqn algorithm with experienced replay
        """

        if len(self.action_history) < self.batch_size:
            
            # replay buffer not large enough to train
            return

        # generate batch indexes
        batch = np.random.randint(
            low = 0,
            high = len(self.action_history),
            size = self.batch_size
        )

        # sample batches
        state_sample = torch.tensor(np.array([self.state_history[b] for b in batch]))
        action_sample = torch.tensor(np.array([self.action_history[b] for b in batch]))
        reward_sample = torch.tensor(np.array([self.rewards_history[b] for b in batch]))
        try:
            a = [self.state_next_history[b] for b in batch]
            state_next_sample = torch.tensor(np.array(a))
            # state_next_sample = torch.tensor(np.array([self.state_next_history[b] for b in batch]))
        except ValueError as err:
            print('state_next error')
            for i in a:
                print(i)
                print(i.shape)
            raise ValueError(err)
        done_sample = torch.tensor(np.array([self.done_history[b] for b in batch]), dtype = torch.float32)

        # print('sarsa batch', state_sample.dtype, action_sample.dtype, reward_sample.dtype, state_next_sample.dtype, done_sample.dtype)

        # create target predictions
        with torch.no_grad():

            Q_next_state = self.target(state_next_sample).view(-1,3,3)

            # print('next state Q', Q_next_state)
            # print(torch.max(Q_next_state, dim = 2).values)
            # print(torch.sum(torch.max(Q_next_state, dim = 2).values, dim = 1))
            # print(torch.sum(torch.max(Q_next_state, dim = 2).values, dim = 1).shape)

            # calculate estimated Q value by taking the maximum Q values of each action decision for the next state and summing across them
            Q_targets = reward_sample + self.gamma * torch.sum(torch.max(Q_next_state, dim = 2).values, dim = 1)

            # add penalty for finished episodes
            Q_targets = Q_targets * (1-done_sample) - self.penalty * done_sample
        
        # print('q next', Q_next_state.dtype)
        # print('q targets', Q_targets.dtype)

        # print('q targets', Q_targets.shape)

        # Nudge weights of trainable variables
        # actions should have shape (3,), so to get relevant actions
        action_sample[:, 1] += 3 # sideways actions have q indexes: 3-5 
        action_sample[:, 2] += 6 # rotation actions have q indexes: 6-8

        # print('action sample', action_sample.dtype)

        relevant_actions = torch.eye(self.out_size)[action_sample]
        relevant_actions = torch.max(relevant_actions, dim = 1).values

        # print('actions', relevant_actions.dtype)

        self.model.zero_grad()
        q_values = self.model(state_sample)

        # print('actions', action_sample)

        # print('rel', relevant_actions)

        # print(relevant_actions.shape)

        # print('q vals', q_values.dtype)

        Q_of_actions = torch.sum(q_values * relevant_actions, dim = 1)

        # print('q actions', Q_of_actions.dtype)

        # print('q of actions', Q_of_actions.shape)
        
        # calculate loss between between principal and target network
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

    def get_running_reward(self) -> float:

        return self.running_reward

    def save_model(self, filename):

        torch.save(self.model.state_dict(), f'./weights/dqn_{filename}.pt')
        print(f'weights saved at /weights/dqn_{filename}.pt')

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
        np.savetxt("./plots/dqn_runningrewards.csv", rr, delimiter=',')
        print('saved running rewards at /plots/dqn_running_rewards.csv')

        plt.plot(np.arange(1, update_steps*len(self.running_rewards)+1, update_steps), self.running_rewards)
        plt.title('running rewards against frame count')
        plt.xlabel('frame count')
        plt.ylabel('running rewards')
        plt.show()
        plt.savefig('./plots/dqn_training_plot.png')
        print('plot saved at /plots/dqn_training_plot.png')