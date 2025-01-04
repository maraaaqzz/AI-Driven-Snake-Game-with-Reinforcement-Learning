import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module): # inherit the module 
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # hidde_size is the output size
        self.linear2 = nn.Linear(hidden_size, output_size)

    # the actual prediction
    def forward(self, x): # x = tensor
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model' #create new dir in the cur path
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gammma):
        self.lr = lr
        self.gamma = gammma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: # one dimension
            # we want : (1, x)
            state = torch.unsqueeze(state, 0) # appends one dimension in the beginning
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state <=> Q = model.predict(state0)
        pred = self.model(state)

        target = pred.clone()

        # 2: Qnew = reward + gamma*max(Q(state1) = (new predicted Q value)) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad() # empty the gradient
        loss = self.criterion(target, pred)
        loss.backward() # back probagation and update the gradient

        self.optimizer.step()
