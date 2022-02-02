from src import *
from utils import *
import math
import random
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


SARS = namedtuple('SARS', ('state_now', 'action', 'reward', 'state_next'))

class ReplayMemory(object):
    
    def __init__(self, maxlen=100000):        
        self.memory = deque(maxlen=maxlen)
        
    def get_batch(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def push(self, SARS):
        self.memory.append(SARS)


class DQN(nn.Module):
    
    def __init__(self, num_inputs, num_actions, device):
        super(DQN, self).__init__()
        
        self.device = device
        
        self.layer1 = nn.Sequential(
            nn.Linear(num_inputs, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        
        self.out = nn.Linear(16, num_actions)
        
        
    def forward(self, x):
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.out(x)
        return x
        

class Agent(object):
    
    def __init__(self, device, M, writer):
        self.device = device
        self.writer = writer
        
        self.inference = False
        
        
        self.BATCH_SIZE = 128
        self.GAMMA = 0.9
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        
        self.num_states = 0
        for m in range(1, M+1):
            self.num_states += m**2
        self.num_states = self.num_states * 2
        self.num_actions = len(constants.ACTION_CODE.values())
        
        self.policy_net = DQN(self.num_states, self.num_actions, self.device).to(self.device)
        self.target_net = DQN(self.num_states, self.num_actions, self.device).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.replay = ReplayMemory(10000)
        self.steps_done = 0
        self.total_count = 0
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
        
    def get_action(self, state):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1 * self.steps_done / self.EPS_DECAY)

        self.steps_done += 1

        if random.random() > eps_threshold:
            self.inference = True
            with torch.no_grad():
                self.policy_net.eval()
                return self.policy_net(state).max(1)[1].view(1,1)
        else:
            self.inference = False
            return torch.tensor([[random.randrange(self.num_actions)]]).to(self.device)
        
        
    def optimize(self):
        if len(self.replay.memory) < self.BATCH_SIZE:
            return
        
        batch = SARS(*zip(*self.replay.get_batch(self.BATCH_SIZE)))
        
        not_none_mask = torch.tensor(tuple(map(lambda x: x is not None, batch.state_next)), device=self.device)
        state_next_batch = torch.cat([x for x in batch.state_next if x is not None]).to(self.device)

        state_batch = torch.cat(batch.state_now).to(self.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        qsa = self.policy_net(state_batch)
        
        
        qsa = qsa.gather(1, action_batch)
        qsa_next = torch.zeros(self.BATCH_SIZE, device=self.device)
        qsa_next[not_none_mask] = self.target_net(state_next_batch).max(1)[0].detach()
        temp_diff_target = ((qsa_next * self.GAMMA) + reward_batch).unsqueeze(1)
        
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(qsa, temp_diff_target)
        
        if self.total_count % 10 == 0:
            self.writer.add_scalar('loss',
                                  loss.item(),
                                  self.total_count)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
