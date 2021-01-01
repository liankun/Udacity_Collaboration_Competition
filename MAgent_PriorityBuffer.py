from model import NetWorkActor, NetWorkCritic
from collections import namedtuple,deque
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from SumTree import SumTree
from DDPGAgent import DDPGAgent

class PriorityReplayPool:
    """
    Fixed-size buffer to store experience tuples
    """
    def __init__(self,buffer_size,batch_size,device='cpu'):
        """
        buffer_size:maximum size of buffer
        batch_size: size of each trainning batch
        """
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.device = device
        
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
    
    def get_priority(self,error):
        return (np.abs(error)+self.e)**self.a
        
    
    def add(self,error,state,action,reward,next_state,done):
        """
        add a new experience to memory
        """    
        p = self.get_priority(error)
        e0 = self.experience(state[0],action[0],reward[0],next_state[0],done[0])
        e1 = self.experience(state[1],action[1],reward[1],next_state[1],done[1])
        self.tree.add(p,(e0,e1))
        
    def sample(self):
        """
        randomly sample a batch of experience from memory according to the
        priority
        """
        #first agent
        batch0 = []
        #second agent
        batch1 = []
        idxs = []
        segment = self.tree.total()/self.batch_size
        priorities = []
        self.beta = np.min([1.,self.beta+self.beta_increment_per_sampling])
        
        for i in range(self.batch_size):
            a = segment*i
            b = segment*(i+1)
            
            s = random.uniform(a,b)
            (idx,p,data) = self.tree.get(s)
            if data==0:
                continue
            priorities.append(p)
            batch0.append(data[0])
            batch1.append(data[1])
            idxs.append(idx)
        
        sampling_probabilities = priorities/self.tree.total()
        is_weight = np.power(self.tree.n_entries*sampling_probabilities,-self.beta)
        is_weight /=is_weight.max()
        
        return (batch0,batch1,idxs,is_weight)
    
        
    def update(self,idx,error):
        p = self.get_priority(error)
        self.tree.update(idx,p)
    
    def __len__(self):
        """
        return the current size of internal memory
        """
        return self.tree.n_entries
        

class MAgent():
    """
    multi-agents
    two players 
    interact with the environment 
    """
    def __init__(self,state_size,
                 action_size,
                 num_agents=2,
                 replay_pool_size=2**17,
                 batch_size=512,
                 gamma=0.95,
                 update_step=4,
                 lr=5e-4,
                 weight_decay=1.e-5,
                 tau=1e-3,
                 add_noise=False,
                 hidden_node1=64*2*2,
                 hidden_node2=32*2*2,
                 device='cpu'):
        """
        state_size:int, the size of state space
        action_size:int, the size of actoin space
        replay_pool_size: the size of replay size need to store
        batch_size: the size of minibatch used in learning
        gamma:discount rate
        weight_decay: weight for the optimizer
        tau: soft update rate
        add_noise: add noise for the action
        hidden_node1: first hidden layer nodes
        hidden_node2: second hidden layer nodes
        update_step: how often to update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.add_noise = add_noise
        self.num_agents=num_agents
        self.buffer_size = replay_pool_size
        
        
        #DDPG Agents
        self.agent0 = DDPGAgent(state_size=state_size,
                        action_size=action_size,
                        num_agents=num_agents,
                        tau=tau,
                        hidden_node1 = hidden_node1,
                        hidden_node2 = hidden_node2,
                        lr = lr,
                        weight_decay = weight_decay,
                        device = device
                        )
        
        self.agent1 = DDPGAgent(state_size=state_size,
                        action_size=action_size,
                        num_agents=num_agents,
                        tau=tau,
                        hidden_node1 = hidden_node1,
                        hidden_node2 = hidden_node2,
                        lr = lr,
                        weight_decay = weight_decay,
                        device = device
                        )
        
        #create replay pool
        self.memory = PriorityReplayPool(replay_pool_size,batch_size,device=self.device)
        
        self.t_step = 0
        self.update_step = update_step
        
    def step(self,state,action,reward,next_state,done):
        #put the experience into the pool
        
        #first agent experience
        state0 = state[0][None,:]
        action0 = action[0][None,:]
        reward0 = reward[0][None,:]
        next_state0 = next_state[0][None,:]
        done0 = done[0][None,:]
        
        state1 = state[1][None,:]
        action1 = action[1][None,:]
        reward1 = reward[1][None,:]
        next_state1 = next_state[1][None,:]
        done1 = done[1][None,:]
        
        with torch.no_grad():
            q0 = self.agent0.critic_forward(torch.from_numpy(state0).float().to(self.device),
                                torch.from_numpy(action0).float().to(self.device),
                                torch.from_numpy(action1).float().to(self.device))
            q1 = self.agent1.critic_forward(torch.from_numpy(state1).float().to(self.device),
                                torch.from_numpy(action1).float().to(self.device),
                                torch.from_numpy(action0).float().to(self.device))
        
            next_action0 = self.agent0.target_act_forward(torch.from_numpy(next_state0).float().to(self.device))
            next_action1 = self.agent1.target_act_forward(torch.from_numpy(next_state1).float().to(self.device))
        
            q_next0 = self.agent0.target_critic_forward(torch.from_numpy(next_state0).float().to(self.device),
                                                        next_action0,next_action1)
            q_next1 = self.agent1.target_critic_forward(torch.from_numpy(next_state1).float().to(self.device),
                                                        next_action1,next_action0)
        
        q_next0 = q_next0.cpu().data.numpy()
        q_next1 = q_next1.cpu().data.numpy()
        
        q_target0 = reward0+self.gamma*q_next0*(1-done0)
        q_target1 = reward1+self.gamma*q_next1*(1-done1)
        
        error0 = q_target0-q_next0
        error1 = q_target1-q_next1
        
        error = np.abs(error0)+np.abs(error1)
        self.memory.add(error[0],state,action,reward,next_state,done)
        
        #learn every update_step
        self.t_step = (self.t_step + 1)%self.update_step
        if self.t_step ==0:
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def act(self,state,eps=0.):
        """
        return actions for given state as per current policy
        state : current state
        eps: scale for the noise
        """
        state = torch.from_numpy(state).float().to(self.device)
        state0 = state[0]
        state1 = state[1]
        self.agent0.actor.eval()
        self.agent1.actor.eval()
        with torch.no_grad():
            action_values0 = self.agent0.act_forward(state0.unsqueeze(0))
            action_values1 = self.agent1.act_forward(state1.unsqueeze(0))
        self.agent0.actor.train()
        self.agent1.actor.train()
        
        #add a normal noise scaled with eps
        action_values0 = action_values0.cpu().data.numpy()
        action_values1 = action_values1.cpu().data.numpy()
        
        #all action values are between -1 and 1
        action_values0 = np.clip(action_values0,-1,1)
        action_values1 = np.clip(action_values1,-1,1)
      
        return np.vstack([action_values0,action_values1])
        
    def learn(self,experiences):
        """
        update the qnetwork local
        experiences: batch0 (first agent experience),batch1(second agent experience),idexs,is_weights
        """
        batch0,batch1,idxs,is_weights = experiences
        
        #put exeriences of agent0 to device
        is_weights = torch.from_numpy(is_weights).float().to(self.device)
        
        states0 = torch.from_numpy(np.vstack([e.state for e in batch0 if e is not None])).float().to(self.device)
        actions0 = torch.from_numpy(np.vstack([e.action for e in batch0 if e is not None])).float().to(self.device)
        rewards0 = torch.from_numpy(np.vstack([e.reward for e in batch0 if e is not None])).float().to(self.device)
        next_states0 = torch.from_numpy(np.vstack([e.next_state for e in batch0 if e is not None])).float().to(self.device)
        dones0 = torch.from_numpy(np.vstack([e.done for e in batch0 if e is not None]).astype(np.uint8)).float().to(self.device)
        
        #put the experience of agent1 to device
        states1 = torch.from_numpy(np.vstack([e.state for e in batch1 if e is not None])).float().to(self.device)
        actions1 = torch.from_numpy(np.vstack([e.action for e in batch1 if e is not None])).float().to(self.device)
        rewards1 = torch.from_numpy(np.vstack([e.reward for e in batch1 if e is not None])).float().to(self.device)
        next_states1 = torch.from_numpy(np.vstack([e.next_state for e in batch1 if e is not None])).float().to(self.device)
        dones1 = torch.from_numpy(np.vstack([e.done for e in batch1 if e is not None]).astype(np.uint8)).float().to(self.device)
        
        q0 = self.agent0.critic_forward(states0,actions0,actions1)
        next_actions0 = self.agent0.target_act_forward(next_states0).detach()
        
        q1 = self.agent1.critic_forward(states1,actions1,actions0)
        next_actions1 = self.agent1.target_act_forward(next_states1).detach()
        
        q_next0 = self.agent0.target_critic_forward(next_states0,next_actions0,actions1)
        q_next1 = self.agent1.target_critic_forward(next_states1,next_actions1,actions0)
        
        q_target0 = rewards0+self.gamma*q_next0*(1-dones0)
        q_target1 = rewards1+self.gamma*q_next1*(1-dones1)
        
        #update the priority
        errors0 = torch.abs(q_target0-q0).cpu().data.numpy()
        errors1 = torch.abs(q_target1-q1).cpu().data.numpy()
        errors = errors0+errors1
        
        #update the priority in the memory
        for i in range(len(idxs)):
            idx = idxs[i]
            self.memory.update(idx,errors[i])
        
        #update the actor and critic of agent0
        critic_loss0 = (is_weights*F.mse_loss(q_target0,q0,reduction='none')).mean()
        self.agent0.critic_optimizer.zero_grad()
        critic_loss0.backward()
        self.agent0.critic_optimizer.step()
        
        
        #update agent1
        critic_loss1 = (is_weights*F.mse_loss(q_target1,q1,reduction='none')).mean()
        self.agent1.critic_optimizer.zero_grad()
        critic_loss1.backward()
        self.agent1.critic_optimizer.step()
        
        del critic_loss0
        del critic_loss1
        
        #optimize actor
        
        pred_actions0 = self.agent0.act_forward(states0)
        pred_actions1 = self.agent1.act_forward(states1)
        
        actor_loss0 = -(is_weights*self.agent0.critic_forward(states0,pred_actions0,actions1)).mean()
        actor_loss1 = -(is_weights*self.agent1.critic_forward(states1,pred_actions1,actions0)).mean()
        
        self.agent0.actor_optimizer.zero_grad()
        actor_loss0.backward()
        self.agent0.actor_optimizer.step()
        
        self.agent1.actor_optimizer.zero_grad()
        actor_loss1.backward()
        self.agent1.actor_optimizer.step()
        
        del actor_loss0
        del actor_loss1
        
        #soft update of target network
        self.soft_update()
        
    def soft_update(self):
        """
        weight_target = tau*weight_local+(1-tau)*weight_target
        """
        #soft update critic
        self.agent0.soft_update()
        self.agent1.soft_update()
        