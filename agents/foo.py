import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from soccer_twos.models.models import LinearModel, CNN

class DeepQN():

    """
    Implementation of Deep Q-Learning algorithm adapted from Deepmind's Deep Q Learning in Atari
    
    Source: https://deepmind.com/research/open-source/dqn
    """

    def __init__(
        self,
        env,
        learning_rate = 0.0005
    ):

        # init environment
        self.env = env

        # init model
        self.model = LinearModel(env.obssize, env.actsize)
        self.target = LinearModel(env.obssize, env.actsize)
        # self.model = CNN(env.obssize, env.actsize)
        # self.target = CNN(env.obssize, env.actsize)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        
    def train(
        self, 
        gamma = 0.99, # discount factor
        batch_size = 64, # batch size
        loss_fn = 'huber',
        max_memory = 1000, # experience replay size
        update_after_actions = 16, # number of actions per model update
        target_update_after_actions = 1000, # number of actions per target model update
        max_episodes = 500, # number of episodes
        max_steps_per_episode = 1000, # number of steps per episode
        last_n_reward = 100, # 
        penalty = 10, # value penalized in target Q-value updates for ended episodes
        target_reward = None, # desired reward for model
        verbose = True, 
        generate_plot = True
    ):

        # init training parameters
        epsilon = np.linspace(1, 0.001, max_episodes) # TODO: modify this to another epsilon scheme?

        
        # initialize replay memory
        state_history = []
        action_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
                
        # tracking training updates
        running_reward = 0
        episode_count = 0
        timestep_count = 0
        running_rewards = []
        
        for episode in range(max_episodes):
            
            state = np.array(self.env.reset())
            episode_reward = 0
            
            for timestep in range(1, max_steps_per_episode):
                
                timestep_count += 1
                
                # exploration
                if np.random.rand() < epsilon[episode]:
                    # Take random action
                    action = self.env.get_random_action()
                else:
                    # Predict action Q-values
                    # From environment state
                    state_t = torch.tensor(state)
                    state_t = torch.unsqueeze(state_t, 0)

                    with torch.no_grad():
                        action_vals = self.model(state_t)

                    # Choose the best action
                    action = torch.argmax(action_vals).item()

                state_next, reward, done, _, _ = self.env.step(action)
                state_next = np.array(state_next)
                episode_reward += reward
                
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                rewards_history.append(reward)
                done_history.append(done)
                
                state = state_next
                
                if timestep_count % update_after_actions == 0 and len(action_history) > batch_size:
                    
                    # generate batch indexes
                    batch = np.random.randint(
                        low = 0,
                        high = len(action_history),
                        size = batch_size
                    )
                    
                    # sample batches
                    state_sample = torch.tensor([state_history[b] for b in batch])
                    state_next_sample = torch.tensor([state_next_history[b] for b in batch])
                    rewards_sample = torch.tensor([rewards_history[b] for b in batch])
                    action_sample = torch.tensor([action_history[b] for b in batch])
                    done_sample = torch.tensor([done_history[b] for b in batch], dtype = torch.int32)
                
                    # create target predictions
                    with torch.no_grad():
                        Q_next_state = self.target(state_next_sample)

                        Q_targets = rewards_sample + gamma * torch.max(Q_next_state, dim = 1)[0]
                    
                        # add penalty for finished episodes
                        Q_targets = Q_targets * (1-done_sample) - penalty * done_sample
                                        
                    # Nudge the weights of the trainable variables towards (considering only relevant actions)
                    relevant_actions = torch.eye(self.env.actsize)[action_sample]
                    relevant_actions = nn.functional.one_hot(action_sample, num_classes = self.env.actsize)
                    
                    self.model.zero_grad()
                    q_values = self.model(state_sample)
                    Q_of_actions = torch.sum(q_values * relevant_actions, dim = 1)
                    
                    # calculate loss between between principal and target network
                    loss = loss_function(Q_targets, Q_of_actions)
                    loss.backward()
                    self.optim.step()
                
                if timestep_count % target_update_after_actions == 0:
                    
                    # update the target network with new weights
                    self.target.load_state_dict(self.model.state_dict())
                    
                    # print results
                    if verbose:                        
                        template = "running reward: {:.2f}, q-values {:.2f} at episode {}, frame count {}, epsilon {}"
                        print(template.format(
                            running_reward, 
                            torch.mean(Q_of_actions), 
                            episode_count, 
                            timestep_count, 
                            epsilon[episode])
                        )
                        running_rewards.append(running_reward)
                        
                if len(rewards_history) > max_memory:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]
                if done:
                    break
            
            # save episode rewards 
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > last_n_reward:
                del episode_reward_history[:1]
            
            running_reward = np.mean(episode_reward_history)
            episode_count += 1
            
            # break if desired reward reached
            if running_reward > target_reward:
                print(f'reward target reached')
                break
        print('training finished with running_reward: ', running_reward)

        if generate_plot:
            plt.plot(np.arange(0, timestep_count, target_update_after_actions)[:-1], running_rewards)
            plt.title('running rewards against frame count')
            plt.xlabel('frame count')
            plt.ylabel('running rewards')
            plt.show()
            # plt.save('./plots/training_plot.png')

    def train(
        self, 
        gamma = 0.99, # discount factor
        batch_size = 64, # batch size
        loss_fn = 'huber',
        max_memory = 1000, # experience replay size
        update_after_actions = 16, # number of actions per model update
        target_update_after_actions = 1000, # number of actions per target model update
        max_episodes = 500, # number of episodes
        max_steps_per_episode = 1000, # number of steps per episode
        last_n_reward = 100, # 
        penalty = 10, # value penalized in target Q-value updates for ended episodes
        target_reward = None, # desired reward for model
        verbose = True, 
        generate_plot = True
    ):

        # init training parameters
        epsilon = np.linspace(1, 0.001, max_episodes) # TODO: modify this to another epsilon scheme?

        # initialize replay memory
        state_history = []
        action_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
                
        # tracking training updates
        running_reward = 0
        episode_count = 0
        timestep_count = 0
        running_rewards = []
        
        for episode in range(max_episodes):
            
            state = np.array(self.env.reset())
            episode_reward = 0
            
            for timestep in range(1, max_steps_per_episode):
                
                timestep_count += 1
                
                # exploration
                if np.random.rand() < epsilon[episode]:
                    # Take random action
                    action = self.env.get_random_action()
                else:
                    # Predict action Q-values
                    # From environment state
                    state_t = torch.tensor(state)
                    state_t = torch.unsqueeze(state_t, 0)

                    with torch.no_grad():
                        action_vals = self.model(state_t)

                    # Choose the best action
                    action = torch.argmax(action_vals).item()

                state_next, reward, done, _, _ = self.env.step(action)
                state_next = np.array(state_next)
                episode_reward += reward
                
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                rewards_history.append(reward)
                done_history.append(done)
                
                state = state_next
                
                if timestep_count % update_after_actions == 0 and len(action_history) > batch_size:
                    
                    # generate batch indexes
                    batch = np.random.randint(
                        low = 0,
                        high = len(action_history),
                        size = batch_size
                    )
                    
                    # sample batches
                    state_sample = torch.tensor([state_history[b] for b in batch])
                    state_next_sample = torch.tensor([state_next_history[b] for b in batch])
                    rewards_sample = torch.tensor([rewards_history[b] for b in batch])
                    action_sample = torch.tensor([action_history[b] for b in batch])
                    done_sample = torch.tensor([done_history[b] for b in batch], dtype = torch.int32)
                
                    # create target predictions
                    with torch.no_grad():
                        Q_next_state = self.target(state_next_sample)
                        Q_targets = rewards_sample + gamma * torch.max(Q_next_state, dim = 1)[0]
                    
                        # add penalty for finished episodes
                        Q_targets = Q_targets * (1-done_sample) - penalty * done_sample
                                        
                    # Nudge the weights of the trainable variables towards (considering only relevant actions)
                    relevant_actions = torch.eye(self.env.actsize)[action_sample]
                    relevant_actions = nn.functional.one_hot(action_sample, num_classes = self.env.actsize)
                    
                    self.model.zero_grad()
                    q_values = self.model(state_sample)
                    Q_of_actions = torch.sum(q_values * relevant_actions, dim = 1)
                    
                    # calculate loss between between principal and target network
                    loss = loss_function(Q_targets, Q_of_actions)
                    loss.backward()
                    self.optim.step()
                
                if timestep_count % target_update_after_actions == 0:
                    
                    # update the target network with new weights
                    self.target.load_state_dict(self.model.state_dict())
                    
                    # print results
                    if verbose:                        
                        template = "running reward: {:.2f}, q-values {:.2f} at episode {}, frame count {}, epsilon {}"
                        print(template.format(
                            running_reward, 
                            torch.mean(Q_of_actions), 
                            episode_count, 
                            timestep_count, 
                            epsilon[episode])
                        )
                        running_rewards.append(running_reward)
                        
                if len(rewards_history) > max_memory:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]
                if done:
                    break
            
            # save episode rewards 
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > last_n_reward:
                del episode_reward_history[:1]
            
            running_reward = np.mean(episode_reward_history)
            episode_count += 1
            
            # break if desired reward reached
            if running_reward > target_reward:
                print(f'reward target reached')
                break
        print('training finished with running_reward: ', running_reward)

        if generate_plot:
            plt.plot(np.arange(0, timestep_count, target_update_after_actions)[:-1], running_rewards)
            plt.title('running rewards against frame count')
            plt.xlabel('frame count')
            plt.ylabel('running rewards')
            plt.show()
            # plt.save('./plots/training_plot.png')
            
    def test(
        self, 
        num_episodes = 100, 
        verbose = True, 
        generate_plot = True
    ):
        
        episode_rewards = []
        for episode in range(100):
            state = np.array(self.env.reset())
            episode_reward = 0

            done = False
            while not done:

                # action
                state_t = torch.tensor(state)
                state_t = torch.unsqueeze(state_t, 0)

                with torch.no_grad():
                    action_vals = self.model(state_t)

                # Choose the best action
                action = torch.argmax(action_vals).item()

                # follow action
                state_next, reward, done, _ , _ = self.env.step(action)
                state_next = np.array(state_next)
                episode_reward += reward
                state = state_next

            # update history
            episode_rewards.append(episode_reward)
            if verbose and (episode+1) % 10 == 0:
                template = "average performance: {} at episode {} "
                print(template.format(np.mean(episode_rewards[-10:]), episode+1))
        print(f'overall average performance: {np.mean(episode_rewards)}')

        if generate_plot:
            plt.plot(np.arange(1, num_episodes+1, 1), episode_rewards)
            plt.title('testing rewards against episodes')
            plt.xlabel('episode')
            plt.ylabel('testing rewards')
            plt.show()
            plt.save('./plots/test_plot.png')

