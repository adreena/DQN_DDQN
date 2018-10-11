import random
from collections import namedtuple, deque
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=["state", "action", "reward", "next_state", "done", "priority"])
        
    def add(self, state, action, next_state, reward, done, priority):
        # todo calculate priority or rank here?
        new_experience = self.experience(state, action, reward, next_state, done, priority)
        self.buffer.append(new_experience)
        
    def random_sample(self, device):
        experiences = random.sample(self.buffer, k=self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        c = 0
        for experience in experiences:
            if experience is not None:
                state_stack = torch.stack(list(experience.state))
                states.append(state_stack)
                next_state_frames = list(experience.state)[1:]
                next_state_frames.append(experience.next_state)
                next_state_stack = torch.stack(next_state_frames)
                next_states.append(next_state_stack)
                
                actions.append(experience.action)
                rewards.append(experience.reward)
                dones.append(experience.done)
                
               
        states = torch.stack(states).float().to(device)
        next_states = torch.stack(next_states).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    
    def __len__(self):
        return len(self.buffer)
