import numpy as np
from q_network import QNetwork
from replay_buffer import ReplayBuffer
import torch
import random
import torch.nn.functional as F

class Agent:
    def __init__(self, env, state_space, action_space, device, learning_rate, buffer_size, \
                 batch_size, gamma, in_channels, train_freq = 4, target_update_freq=1e4, is_ddqn = False):
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        
        self.QNetwork_local = QNetwork(in_channels, self.state_space, self.action_space.n, device = device).to(device)
        self.QNetwork_local.init_weights()
        self.QNetwork_target = QNetwork(in_channels, self.state_space,self.action_space.n , device = device).to(device)
        self.QNetwork_target.load_state_dict(self.QNetwork_local.state_dict())
        
        self.optimizer = torch.optim.RMSprop(self.QNetwork_local.parameters(), lr=learning_rate, alpha=0.95, eps=0.01, centered=True)
        self.criterion = torch.nn.MSELoss()
        self.memory = ReplayBuffer(capacity=int(buffer_size), batch_size=batch_size)
        self.step_count = 0.
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.device = device
        self.buffer_size = buffer_size
        self.num_train_updates = 0
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq

        self.is_ddqn = is_ddqn
        
        print('Agent initialized with memory size:{}'.format(buffer_size))
        
    def act(self, state, epsilon):
        if random.random() > epsilon:
            #switch to evaluation mode, evaluate state, switch back to train mode
            self.QNetwork_local.eval()
            if torch.no_grad():
                actions = self.QNetwork_local(state)
            self.QNetwork_local.train()
            
            actions_np = actions.data.cpu().numpy()[0]
            best_action_idx = int(np.argmax(actions_np))
            return best_action_idx
        else:
            rand_action = self.action_space.sample()
            return rand_action
            
        
    def step(self, state, action, reward, next_state , done, add_memory=True):
        #TODO calculate priority here?
        if add_memory: 
            priority = 1.
            reward_clip = np.sign(reward)
            self.memory.add(state=state, action=action, next_state=next_state, reward=reward_clip, done=done, priority=priority)
            self.step_count = (self.step_count+1) % self.train_freq  #self.update_rate
            
            #if self.step_count == 0 and len(self.memory) >= self.batch_size:
            self.network_is_updated = False
            if self.step_count == 0 and len(self.memory) == self.buffer_size:
                samples = self.memory.random_sample(self.device)
                self.learn(samples)
                self.num_train_updates +=1
                self.network_is_updated = True
            
    def learn(self, samples):
        states, actions, rewards, next_states, dones = samples
        

        if self.is_ddqn is True:
            # DDQN: find max action using local network & gather the values of actions from target network
            next_actions = torch.argmax(self.QNetwork_local(next_states).detach(), dim=1).unsqueeze(1)
            q_target_next = self.QNetwork_target(next_states).gather(1,next_actions)
        else:
            # DQN: find the max action from target network
            q_target_next = self.QNetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    
        # expected actions
        q_local_current = self.QNetwork_local(states).gather(1,actions)

        self.optimizer.zero_grad() #cleans up previous values

        # TD Error 
        TD_target = rewards + (self.gamma*q_target_next * (1-dones))
        TD_error = self.criterion(q_local_current, TD_target)
        TD_error.backward()
        torch.nn.utils.clip_grad_norm_(self.QNetwork_local.parameters(), 5.)
        self.optimizer.step()
        
        if (self.num_train_updates/self.train_freq) % self.target_update_freq == 0:
            self.QNetwork_target.load_state_dict(self.QNetwork_local.state_dict())
        
