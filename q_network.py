import torch
import torch.nn.functional as F 
from torch.autograd import Variable

class QNetwork(torch.nn.Module):
    def __init__(self, in_channels, state_space, action_space, seed = 1, device='cpu'):
        super(QNetwork,self).__init__()
        self.device = device
        self.seed = seed
        torch.manual_seed(self.seed)
        self.action_space = action_space
        self.state_space = state_space
        
        self.conv1 = torch.nn.Conv2d(in_channels, 32, 8, stride=4) # 84-8+4/4 -> 20x20x32
        self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2) # 20-4+2/2 -> 9x9x64
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1) # 9-3+1/1 -> 7x7x64
        self.fc1 = torch.nn.Linear(7*7*64, 512) 
        self.fc2 = torch.nn.Linear(512, action_space)
        
    
    def init_weights(self):
        self.conv1.weight.data.uniform_(-0.1, 0.1)
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        self.conv3.weight.data.uniform_(-0.1, 0.1)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        print('weights initialized')
        
    def forward(self, state):
        state = state.to(self.device)
        output = F.relu(self.conv1(state))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = output.view(output.size(0),-1)
        output = F.relu(self.fc1(output))
        return self.fc2(output)
