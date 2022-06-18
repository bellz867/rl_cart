import math
import random

import gym
print(gym.__file__)
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from cartpole_env_ppo import CartPoleEnv
from torch.distributions import Normal
import torch.nn.functional as F

import os
import datetime

from IPython.display import clear_output
import matplotlib.pyplot as plt

from collections import OrderedDict

import multiprocessing_env
from multiprocessing_env import SubprocVecEnv

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# create a class wrapper from PyTorch nn.Module to use exp
class Exp(nn.Module):
    """
    Applies exponential radial basis \n
    Returns:
    -------
    \t Exp(x) = exp(-x**2) \n
    """
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(-x**2)

# create a class wrapper from PyTorch nn.Module to use x^2
class Sqr(nn.Module):
    """
    Applies square basis \n
    Returns:
    -------
    \t Sqr(x) = x**2 \n
    """
    def __init__(self):
        super(Sqr, self).__init__()

    def forward(self, x):
        return x**2

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.001):
        super(ActorCritic, self).__init__()
        
        self.head = nn.Sequential(OrderedDict([
            ('li',nn.Linear(num_inputs,hidden_size)),
            ('ai',nn.Tanh()),
            ('lo',nn.Linear(hidden_size, hidden_size)),
            ('ao',nn.Mish())])
        )

        self.critic = nn.Sequential(OrderedDict([
            ('ai',Sqr()),
            ('li',nn.Linear(hidden_size, hidden_size)),
            ('ao',nn.Mish()),
            ('lo',nn.Linear(hidden_size, 1))])
        )
        
        self.actor_mean = nn.Sequential(OrderedDict([
            ('ai',Exp()),
            ('li',nn.Linear(hidden_size, hidden_size)),
            ('ai2',Exp()),
            ('lo',nn.Linear(hidden_size, 1)),
            ('ao',nn.Tanh())])
        )

        self.actor_log_std = nn.Sequential(OrderedDict([
            ('ai',Exp()),
            ('li',nn.Linear(hidden_size, hidden_size)),
            ('ai2',Exp()),
            ('lo',nn.Linear(hidden_size, 1)),
            ('ao',Exp())])
        )
        self.actor_log_std_scale = nn.Parameter(torch.ones(1)*std)
        
    def forward(self, x):
        head = self.head(x)
        value = self.critic(head)
        mean = self.actor_mean(head)
        std = (self.actor_log_std_scale*self.actor_log_std(head)).exp()
        return mean, std, value


# Hyper params:
hidden_size  = 32     # hidden size
lr           = 0.0005 # learning rate
actor_gain   = 0.1    # actor loss gain
critic_gain  = 0.01    # critic loss gain
entropy_gain = 0.1  # entropy loss gain
num_cycles   = 2   # number of cycles of batches
num_run_bchs = 10     # number of batches per run
num_envs     = 10     # number of environments per cycle
num_trn_stps = 20     # number of steps to use each training per run
num_tst_stps = 200    # number of steps per run
num_epochs   = 32     # number of epochs per batch
mini_bch_size= 16    # size of minibatches per epoch
clip_eps     = 0.1    # clip ratio value bound
frame_idx    = 0      # total number of steps
test_rewards = []

# weights and losses buffers for plotting
head_lis = []
head_los = []
mean_lis = []
mean_los = []
std_lis = []
std_los = []
critic_lis = []
critic_los = []
actor_losses = []
critic_losses = []
entropy_losses = []
std_scales = []

def plot(data, name, path):
    clear_output(True)
    plt.figure(figsize=(10,5))
    plt.title(name)
    plt.plot(data)
    plt.savefig(path+"/"+name+".pdf")
    
def test_env(env, vis=False,cycle_idx=0):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0.0
    for step in range(num_tst_stps):
        state = torch.tensor(state,device=device)
        act_mean, act_std, value = model(state)
        dist  = Normal(act_mean,act_std)
        # action = dist.sample()
        action = dist.mean.detach()
        # print("state: " +str(state))
        # print("action: " +str(action))
        action = torch.clip(action,-1.0,1.0).cpu().numpy().item()
        # print("action clip: " +str(action))
        # print("state: " +str(state))
        next_state, reward, _, _ = env.step(action)
        print("angle: "+str(round(state[2].item(),3))+" dist mean: " +str(round(dist.mean.item(),3))+" dist std: " +str(round(dist.stddev.item(),3))+" action: "+str(round(action,3))+" reward: "+str(round(reward,3)))
        state = next_state
        if vis: env.render()
        total_reward += reward
    print("cycle "+str(cycle_idx)+" total reward "+str(total_reward))
    return total_reward

# def compute_returns(next_value, rewards, masks, gamma=0.99):
#     R = next_value
#     returns = []
#     for step in reversed(range(len(rewards))):
#         R = rewards[step] + gamma * R * masks[step]
#         returns.insert(0, R)
#     return returns

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# for whatever reason this must be done to make the environments permanent in the list if you initialize 
# def make_perm_env():
#     def create_env():
#         env = CartPoleEnv()
#         return env
#     return create_env

if __name__ == "__main__":
    # model for ppo
    model = ActorCritic(4, 1, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    # environment to test
    env = CartPoleEnv()

    # environments to train
    envs = []
    for ii in range(num_envs):
        envs.append(CartPoleEnv)
    # print(envs)
    # envs = [make_perm_env() for ii in range(num_envs)]
    envs = SubprocVecEnv(envs)
    
    # loop the training cycles
    for cycle_idx in range(num_cycles):

        state = envs.reset()
        # print (state)

        # loop the run for number of batches
        for batch_idx in range(num_run_bchs):
            # reset all the training storage for the next batch
            log_prob_batch = []
            values_batch = []
            states_batch = []
            actions_batch = []
            rewards_batch = []
            masks_batch = []
            returns_batch = []

            # loop through a new batch
            for step in range(num_trn_stps):
                # get the distribution for the current state 
                # sample an action from the distribution 
                # then calculate the log of the probability of that action
                # and the entropy of the distribution
                state = torch.tensor(state,device=device)
                act_mean, act_std, value = model(state)
                dist  = Normal(act_mean,act_std)
                action = dist.sample().squeeze()
                # print("state: " +str(state))
                # print("action: " +str(action))
                action = torch.clip(action,-1.0,1.0)
                # print("action clip: " +str(action))
                log_prob = dist.log_prob(action.view(num_envs,1))

                # step the envs using the sample action then calculate the log of the probability
                next_state, reward, done, _ = envs.step(action.cpu().numpy())

                # print("next_state: " +str(next_state))
                # print("log_prob "+str(log_prob))
                # print("entropy_batch "+str(entropy_batch))
                # print("value "+str(value))
                # print("reward "+str(torch.tensor(reward,device=device).unsqueeze(1)))
                # print("masks "+str(torch.tensor(1 - done,device=device).unsqueeze(1)))

                # save the step information to the batch
                log_prob_batch.append(log_prob.view(num_envs,1))
                values_batch.append(value)
                rewards_batch.append(torch.tensor(reward,device=device).unsqueeze(1))
                masks_batch.append(torch.tensor(1 - done,device=device).unsqueeze(1))
                states_batch.append(state)
                actions_batch.append(action.view(num_envs,1))

                # copy over the state and index frames
                state = next_state
                frame_idx += 1
            
            next_state = torch.tensor(next_state,device=device)
            next_mean, next_variance, next_value = model(next_state)
            returns_batch = compute_gae(next_value,rewards_batch,masks_batch,values_batch)

            
            # reshape batch into single column
            log_prob_batch = torch.cat(log_prob_batch).detach()
            returns_batch = torch.cat(returns_batch).detach()
            values_batch = torch.cat(values_batch).detach()
            states_batch = torch.cat(states_batch)
            actions_batch = torch.cat(actions_batch)
            advantages_batch = returns_batch - values_batch
            
            # run number of epochs on the data
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            entropy_loss_sum = 0.0
            avg_idx = 0
            for epoch in range(num_epochs):
                # generate random set of mini batch from the batch
                mini_batch_start = np.arange(0,num_trn_stps*num_envs,mini_bch_size,dtype=np.int64)
                indices = np.arange(num_trn_stps*num_envs,dtype=np.int64)
                np.random.shuffle(indices)
                np.random.shuffle(mini_batch_start)
                mini_batches = [indices[ii:ii+mini_bch_size] for ii in mini_batch_start]
                # print(mini_batches)
                # print(states_batch)
                # print(actions_batch)

                for batch in mini_batches:
                    if len(batch) == mini_bch_size:
                        # get the datat for the mini batch
                        log_prob_old_batchi = log_prob_batch[batch,:]
                        returns_batchi = returns_batch[batch,:]
                        states_batchi = states_batch[batch,:]
                        actions_batchi = actions_batch[batch,:]
                        advantages_old_batchi = advantages_batch[batch,:]

                        # get model, log probs, and entropy
                        act_mean_batchi, act_std_batchi, values_batchi = model(states_batchi)
                        dist_batchi  = Normal(act_mean_batchi,act_std_batchi)
                        log_prob_batchi = dist_batchi.log_prob(actions_batchi)
                        advantages_batchi = returns_batchi - values_batchi
                        entropy_batchi = dist_batchi.entropy().mean()

                        # calculate loss
                        ratio_batchi = (log_prob_batchi-log_prob_old_batchi).exp()
                        surrogate1_batchi = ratio_batchi*advantages_old_batchi
                        surrogate2_batchi = torch.clip(ratio_batchi,1.0-clip_eps,1.0+clip_eps)*advantages_old_batchi
                        actor_loss  = -actor_gain*torch.min(surrogate1_batchi,surrogate2_batchi).mean()
                        critic_loss = critic_gain*advantages_batchi.pow(2).mean()
                        entropy_loss = -entropy_gain*entropy_batchi
                        loss = actor_loss+critic_loss+entropy_loss

                        # print("critic loss "+str(critic_loss))
                        # print("entropy loss "+str(entropy_batch))
                        # print("loss "+str(loss))
                        # print("params critic l1 b \n"+str(model.critic.l1.weight))
                        # print("params critic l2 b \n"+str(model.critic.l2.weight))
                        # print("params actor l1 b \n"+str(model.actor.l1.weight))
                        # print("params actor l2 b \n"+str(model.actor.l2.weight))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        actor_loss_sum += actor_loss.cpu().detach().numpy()
                        critic_loss_sum += critic_loss.cpu().detach().numpy()
                        entropy_loss_sum += entropy_loss.cpu().detach().numpy()
                        avg_idx += 1
                # print("epoch: "+str(epoch)+" critic loss: "+str(round(critic_loss_sum/avg_idx,4))+" actor loss: "+str(round(actor_loss_sum/avg_idx,4))+" entropy: "+str(round(entropy_batch_sum/avg_idx,4)))
            print("cycle: "+str(cycle_idx)+" batch: "+str(batch_idx)+" critic loss: "+str(round(critic_loss_sum/avg_idx,4))+" actor loss: "+str(round(actor_loss_sum/avg_idx,4))+" entropy: "+str(round(entropy_loss_sum/avg_idx,4)))
            head_lis.append(model.head.li.weight.detach().cpu().view(-1,).numpy())
            head_los.append(model.head.lo.weight.detach().cpu().view(-1,).numpy())
            mean_lis.append(model.actor_mean.li.weight.detach().cpu().view(-1,).numpy())
            mean_los.append(model.actor_mean.lo.weight.detach().cpu().view(-1,).numpy())
            std_lis.append(model.actor_log_std.li.weight.detach().cpu().view(-1,).numpy())
            std_los.append(model.actor_log_std.lo.weight.detach().cpu().view(-1,).numpy())
            critic_lis.append(model.critic.li.weight.detach().cpu().view(-1,).numpy())
            critic_los.append(model.critic.lo.weight.detach().cpu().view(-1,).numpy())
            actor_losses.append(actor_loss_sum/avg_idx)
            critic_losses.append(critic_loss_sum/avg_idx)
            entropy_losses.append(entropy_loss_sum/avg_idx)
            std_scales.append(model.actor_log_std_scale.detach().cpu().numpy())

        # test after training for an entire run
        test_rewards.append(np.mean([test_env(env,vis=True,cycle_idx=cycle_idx) for _ in range(1)]))
    

    #start save file
    now = datetime.datetime.now()
    nownew = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = "ppo-sim-"+nownew
    os.makedirs(path)
    plot(test_rewards,'test reward',path)
    plot(head_lis,'head input weights',path)
    plot(head_los,'head output weights',path)
    plot(mean_lis,'actor mean input weights',path)
    plot(mean_los,'actor mean output weights',path)
    plot(std_lis,'actor log std input weights',path)
    plot(std_los,'actor log std output weights',path)
    plot(critic_lis,'critic input weights',path)
    plot(critic_los,'critic output weights',path)
    plot(actor_losses,'actor loss',path)
    plot(critic_losses,'critic loss',path)
    plot(entropy_losses,'entropy',path)
    plot(std_scales,'log std scale',path)
        