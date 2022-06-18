import math
import random

import gym
print(gym.__file__)
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from cartpole_env_ac3_discrete import CartPoleEnv
from torch.distributions import Categorical
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
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(OrderedDict([
            ('l1',nn.Linear(num_inputs,hidden_size)),
            ('a11',nn.Tanh()),
            ('a12',Sqr()),
            ('l2',nn.Linear(hidden_size, hidden_size)),
            ('a2',nn.ReLU()),
            ('l3',nn.Linear(hidden_size, 1))])
        )
        
        self.actor = nn.Sequential(OrderedDict([
            ('l1',nn.Linear(num_inputs, hidden_size)),
            ('a1',Exp()),
            ('l2',nn.Linear(hidden_size, hidden_size)),
            ('a2',Exp()),
            ('l3',nn.Linear(hidden_size, num_outputs)),
            ('a3',nn.Tanh())])
        )
        
    def forward(self, x):
        # print("x "+str(x))
        value = self.critic(x)
        policy = self.actor(x)
        # print("value "+str(value))
        # print("policy "+str(policy))
        return policy, value


# Hyper params for A3C discrete out:
hidden_size  = 128    # hidden size
lr           = 0.0001 # learning rate
actor_gain   = 1.0   # actor loss gain
critic_gain  = 0.1   # critic loss gain
entropy_gain = 10.0   # entropy loss gain
num_cycles   = 100   # number of cycles of batches
num_run_bchs = 16    # number of batches per run
num_envs     = 5    # number of environments per cycle
num_trn_stps = 8    # number of steps to use each training per run
sample_size  = num_trn_stps*num_envs
num_tst_stps = 200   # number of steps per run
frame_idx    = 0     # total number of steps
test_rewards = []
actor_l1s = []
critic_l1s = []
actor_losses = []
critic_losses = []
entropy_losses = []


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
        policy, value = model(state)
        probs = F.softmax(policy,dim=0)
        dist  = Categorical(probs)
        action = dist.sample()
        # print("state: " +str(state))
        next_state, reward, _, _ = env.step(action.cpu().numpy())
        print("angle: "+str(round(state[2].item(),3))+" dist: " +str(dist.probs)+" action: "+str(action.item())+" reward: "+str(round(reward,3)))
        state = next_state
        if vis: env.render()
        total_reward += reward
    print("cycle "+str(cycle_idx)+" total reward "+str(total_reward))
    return total_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

# for whatever reason this must be done to make the environments permanent in the list if you initialize 
# def make_perm_env():
#     def create_env():
#         env = CartPoleEnv()
#         return env
#     return create_env

if __name__ == "__main__":
    # model for A3C
    model = ActorCritic(4, 2, hidden_size).to(device)
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
        for batch in range(num_run_bchs):
            # reset all the training storage for the next batch
            log_prob_batch = []
            values_batch = []
            rewards_batch = []
            masks_batch = []
            returns_batch = []
            entropy_batch = 0

            # loop through a new batch
            for step in range(num_trn_stps):
                # get the distribution for the current state 
                # sample an action from the distribution 
                # then calculate the log of the probability of that action
                # and the entropy of the distribution
                state = torch.tensor(state,device=device)
                policy, value = model(state)
                probs = F.softmax(policy,dim=1)
                dist  = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy_batch += dist.entropy().mean()
                
                print("state: " +str(state))
                print("policy: " +str(policy))
                print("probs: " +str(probs))
                print("dist: " +str(dist))
                print("action: " +str(action))
                

                # step the envs using the sample action then calculate the log of the probability
                next_state, reward, done, _ = envs.step(action.cpu().numpy())

                print("next_state: " +str(next_state))
                # print("log_prob "+str(log_prob))
                # print("entropy_batch "+str(entropy_batch))
                # print("value "+str(value))
                # print("reward "+str(torch.tensor(reward,device=device).unsqueeze(1)))
                # print("masks "+str(torch.tensor(1 - done,device=device).unsqueeze(1)))

                # save the step information to the batch
                log_prob_batch.append(log_prob)
                values_batch.append(value)
                rewards_batch.append(torch.tensor(reward,device=device).unsqueeze(1))
                masks_batch.append(torch.tensor(1 - done,device=device).unsqueeze(1))
                
                # copy over the state and index frames
                state = next_state
                frame_idx += 1
            
            next_state = torch.tensor(next_state,device=device)
            _, next_value = model(next_state)
            returns_batch = compute_returns(next_value,rewards_batch,masks_batch)
            
            # reshape batch into single column
            log_prob_batch = torch.cat(log_prob_batch)
            returns_batch = torch.cat(returns_batch).detach()
            values_batch = torch.cat(values_batch)
            
            #caluclate advantage
            advantages_batch = returns_batch - values_batch
            
            # average entropy
            entropy_batch /= sample_size

            # calculate loss
            actor_loss  = -(log_prob_batch*advantages_batch.detach()).mean()
            critic_loss = advantages_batch.pow(2).mean()
            loss = actor_gain*actor_loss + critic_gain*critic_loss - entropy_gain*entropy_batch

            print("total loss: "+str(round(loss.item(),4))+" critic loss: "+str(round(critic_loss.item(),4))+" actor loss: "+str(round(actor_loss.item(),4))+" entropy: "+str(round(entropy_batch.item(),4)))
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
            
            # print("params critic l1 a \n"+str(model.critic.l1.weight))
            # print("params critic l2 a \n"+str(model.critic.l2.weight))
            # print("params actor l1 a \n"+str(model.actor.l1.weight))
            # print("params actor l2 a \n"+str(model.actor.l2.weight))

            actor_l1s.append(model.actor.l1.weight.detach().cpu().view(-1,).numpy())
            critic_l1s.append(model.critic.l1.weight.detach().cpu().view(-1,).numpy())
            actor_losses.append(actor_loss.cpu().detach().numpy())
            critic_losses.append(critic_loss.cpu().detach().numpy())
            entropy_losses.append(entropy_batch.cpu().detach().numpy())

        # test after training for an entire run
        test_rewards.append(np.mean([test_env(env,vis=True,cycle_idx=cycle_idx) for _ in range(1)]))
    
    #start save file
    now = datetime.datetime.now()
    nownew = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = "ac3-d-sim-"+nownew
    os.makedirs(path)
    plot(test_rewards,'test reward',path)
    plot(actor_l1s,'actor weights',path)
    plot(critic_l1s,'critic weights',path)
    plot(actor_losses,'actor loss',path)
    plot(critic_losses,'critic loss',path)
    plot(entropy_losses,'entropy',path)
        