import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import time

from soccer_twos.models.models import LinearModel, CNN
from soccer_twos.environment import SoccerTwosEnv


class VPG():
    
    def __init__(
        self,
        env, 
        actor_lr = 4e-3, 
        critic_lr = 2e-3
        ):
        
        # init env
        self.env = env
        
        # init models
        self.actor = LinearModel(self.env.obssize, self.env.actsize)
        self.critic = LinearModel(self.env.obssize, 1)

        # init model optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

    def train(
        self,
        n_episodes = 1500, # number of training episodes
        gamma = 0.99, # discount factor
        last_n_reward = 100, # num of episodes in running reward calculation
        batch_size = 5000, # number of trajectories per episode
        target_reward = -200, # desired running reward
        consistency_target = 10, # desired consistency rate for target_reward
        verbose = True, # print episode logs
        plot_rewards = True, # plot running rewards after training,
        save_model = True
    ):
        # init training parameters
        self.gamma = gamma

        # init training logs
        episode_reward_history = []
        running_rewards = []
        initial_time = time.time()

        for episode in range(n_episodes):

            start_time = time.time()

            # sample trajectories using current policy
            states, actions, rewards, n_dones, episode_reward = self.sample_traj(
                batch_size = batch_size
            )
            
            # update actor and critic models using samples
            self.fit(states, actions, rewards, n_dones)

            # save reward
            episode_reward_history.append(episode_reward)

            # keep only last n rewards (to save memory)
            if len(episode_reward_history) > last_n_reward:
                del episode_reward_history[0]

            running_reward = np.mean(episode_reward_history)
            running_rewards.append(running_reward)

            # track running time
            end_time = time.time()
            episode_runtime = end_time - start_time
            total_runtime = end_time - initial_time

            # print logging information
            if verbose:
                print_info = [
                    f"Episode: {episode + 1}",
                    f"Reward: {episode_reward:.3f}, Running R.: {running_reward:.3f}",
                    f"Runtime: {episode_runtime:.2f}s, Total T.: {total_runtime:.2f}s"
                ]
                print(", ".join(print_info))
            
            # save actor model if reward target met consistently
            if episode_reward >= target_reward:
                consistency_count += 1
                if consistency_count >= consistency_target:
                    if save_model:
                        torch.save(self.actor.state_dict(), './actor_weights.pt')
                    print('Consistency target reached, actor saved')
                    break
            else:
                consistency_count = 0
        
        if plot_rewards:
            plt.plot(running_rewards)
            plt.title('running rewards against episodes')
            plt.xlabel('episode')
            plt.ylabel('running rewards')
            plt.show()
            
    def sample_traj(self, batch_size = 2000, seed = None):
        
        # init trajectory buffer
        states = []
        actions = []
        rewards = []
        not_dones = []
        curr_reward_list = []

        while len(states) < batch_size:
            # reset env
            state = self.env.reset(seed = seed)
            curr_reward = 0
            done = False

            # step through env with action samples from actor
            while not done:
                
                state_tensor = torch.unsqueeze(
                    torch.tensor(state, dtype = torch.float32),
                    dim = 0
                )

                # sample action
                with torch.no_grad():
                    action = self.env.sample_action(
                        self.actor(state_tensor)
                    )

                # take step in env
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # check if episode done
                done = terminated or truncated

                # store current state, action, reward, done
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                not_dones.append(not done)

                # prepare next step
                state = next_state
                curr_reward += reward

                if done:
                    break
        
            curr_reward_list.append(curr_reward)

        return np.array(states), np.array(actions), np.array(rewards), np.array(not_dones), np.mean(curr_reward_list)

    def fit(self, states, actions, rewards, n_dones):

        states = torch.tensor(states, dtype = torch.float32)
        actions = torch.tensor(actions, dtype = torch.float32)
        rewards = torch.tensor(rewards, dtype = torch.float32)
        n_dones = torch.tensor(n_dones, dtype = torch.float32)
        
        # calculate discounted rewards and number of trajectories in episode sample
        G_t, num_traj = self.discount_rewards(rewards, n_dones)

        # compute critic loss
        critics = self.critic(states)
        critic_loss = torch.sum(torch.square(G_t-torch.squeeze(critics)))

        # update critic
        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        means, log_stds = self.actor(states)
        # print(means, log_stds)
        stds = torch.exp(log_stds)

        # calculate log probability
        neg_log_prob = 0.5 * torch.sum(torch.square((actions - means) / (stds + 1e-8)), dim = 1)
        neg_log_prob += 0.5 * torch.log(torch.tensor(2.0 * np.pi))
        neg_log_prob += torch.sum(log_stds, dim = 1)

        # compute and normalize advantages
        advantages = G_t - critics.detach()
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

        # compute actor loss
        self.actor.zero_grad()
        actor_loss = torch.sum(advantages*neg_log_prob)/num_traj
        
        # update actor
        self.actor.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
    
    def discount_rewards(self, reward_buffer, n_dones):

        G_t = torch.zeros_like(reward_buffer, dtype = float)
        running_add = 0
        num_traj = 0

        for t in reversed(range(len(reward_buffer))):
            # if end of episode, reset accumulator and increment traj count
            if n_dones[t] == 0:
                running_add = 0
                num_traj += 1
            running_add = reward_buffer[t] + self.gamma*running_add
            G_t[t] = running_add
        
        return G_t, num_traj