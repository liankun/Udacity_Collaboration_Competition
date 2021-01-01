from model import NetWorkActor, NetWorkCritic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy


class OUNoise:
    """
    Ornstein-Uhlenbeck process
    """
    def __init__(self,size,seed,mu=0.,theta=0.15,sigma=0.2):
        """initialize"""
        self.mu=mu*np.ones(size)
        self.theta = theta
        self.sigma=sigma
        self.seed = random.seed(seed)
        self.state = copy.copy(self.mu)
    
    def sample(self):
        "generate noise"
        x = self.state
        dx = self.theta*(self.mu-x)+self.sigma*np.array([random.random() for i in range(len(x))])
        self.state = x+dx
        
        return self.state


class DDPGAgent:
    def __init__(self,state_size,action_size,num_agents,tau,hidden_node1,hidden_node2,lr,weight_decay,device='cpu'):
        """
        state_size: the size of state space in this task , it is 24
        action_size: the size of action space
        num_agents: number of agents
        tau: soft update parameter
        hidden_node1:the nodes of first hidden layer
        hidden_node2:the nodes of second hidden layer
        lr:learning rate
        weight_decay: the weight decay for the optimizer
        """
        super(DDPGAgent,self).__init__()
        
        self.device = device
        self.tau=tau
        self.actor = NetWorkActor(state_size=state_size,action_size=action_size,
                          hidden_node1=hidden_node1,hidden_node2=hidden_node2).to(device)
        self.target_actor = NetWorkActor(state_size=state_size,action_size=action_size,
                               hidden_node1=hidden_node1,hidden_node2=hidden_node2).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = NetWorkCritic(state_size=state_size,action_size=action_size,num_agents=num_agents,
                           hidden_node1=hidden_node1,hidden_node2=hidden_node2).to(self.device)
        self.target_critic = NetWorkCritic(state_size=state_size,action_size=action_size,num_agents=num_agents,
                           hidden_node1=hidden_node1,hidden_node2=hidden_node2).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=lr,weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=lr)
        
        
    def act_forward(self,state):
        action = self.actor.forward(state)
        return action
    
    
    def target_act_forward(self,state):
        action = self.target_actor.forward(state)
        return action
    
    def critic_forward(self,state,action,other_action):
        val = self.critic.forward(state,torch.cat((action,other_action),dim=1))
        return val
    
    def target_critic_forward(self,state,action,other_action):
        val = self.target_critic.forward(state,torch.cat((action,other_action),dim=1))
        
        return val
        
    def soft_update(self):
        """
         weight_target = tau*weight_local+(1-tau)*weight_target
        """
        for target_param, param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        
        #soft update actor
        for target_param, param in zip(self.target_actor.parameters(),self.actor.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        
        
        
        