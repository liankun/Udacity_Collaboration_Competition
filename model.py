import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    Used for parameter initialization
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class NetWorkActor(nn.Module):
    """
    The network for the actor
    """
    def __init__(self,state_size,action_size,hidden_node1=64*2*2,hidden_node2=32*2*2):
        """
        input_size: int
        output_size: int
        hidden_layers: a list of int
        is_actor: for the actor, the last layer will use the tanh
        """
        super(NetWorkActor,self).__init__()
        self.fc1 = nn.Linear(state_size,hidden_node1)
        self.fc2 = nn.Linear(hidden_node1,hidden_node2)
        self.fc3 = nn.Linear(hidden_node2,action_size)
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #make sure action value is from -1 to 1
        x = torch.tanh(self.fc3(x))
        return x
    


#model 
class NetWorkCritic(nn.Module):
    """
    The network for the actor
    """
    def __init__(self,state_size,action_size,num_agents,hidden_node1=64*2*2,hidden_node2=32*2*2):
        """
        input_size: int
        output_size: int
        hidden_layers: a list of int
        is_actor: for the actor, the last layer will use the tanh
        """
        super(NetWorkCritic,self).__init__()
        self.fc1 = nn.Linear(state_size,hidden_node1)
        self.fc2 = nn.Linear(hidden_node1+num_agents*action_size,hidden_node2)
        self.fc3 = nn.Linear(hidden_node2,1)
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self,state,actions):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat((x,actions),dim=1)))
        x = self.fc3(x)
        return x