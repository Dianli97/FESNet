from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import config
from model import I2CNet
from dataset import EMGDataset
from preprocessor import preprocess_data
import pickle
import time
import pandas as pd


batch_size = config.batch_size
epochs = config.num_epochs
learning_rate = config.learning_rate
window_size = config.window_size  
train_path = 'XX' #PATH TO YOUR TRAIN DATASET
loss_log = []
width = config.width
acc = config.acc

if config.width != 1:
    task = 'class'
else:
    task = 'seg'

subj_list = [0,1,2,3,4,5,6,7,8,9,10,11]
# subj_list = [6]
ex_list = [0]
acq_list = [0, 1]
# x_data, y_data, scaler_x, scaler_y = preprocess_data(subj_list, ex_list, acq_list, window_size)
x_data, y_data = preprocess_data(train_path, subj_list, ex_list, acq_list, window_size)
# print(f'Original labels - Min: {y_data.min().item()}, Max: {y_data.max().item()}')

# with open('scaler_x.pkl', 'wb') as f:
#     pickle.dump(scaler_x, f)
# with open('scaler_y.pkl', 'wb') as f:
#     pickle.dump(scaler_y, f)

dataset = EMGDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class CombinedLoss(nn.Module):
    def __init__(self, weight_pearson=0.5, weight_mae=0.5, weight_ce=0.5):
        super(CombinedLoss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_pearson = weight_pearson
        self.weight_mae = weight_mae
        self.weight_ce = weight_ce

    def forward(self, outputs, targets):
        # To [batchsize * channels * length, num_class]
        outputs_reshaped = outputs.permute(0, 1, 3, 2).contiguous().view(-1, outputs.size(2))
        # To [batchsize * channels * length]
        targets_reshaped = targets.view(-1).long()

        ce = self.ce_loss(outputs_reshaped, targets_reshaped)
        
        outputs_max = outputs.argmax(dim=2)
        
        mae = self.mae_loss(outputs_max.float(), targets.float())
        
        vx = outputs_max.float() - torch.mean(outputs_max.float())
        vy = targets.float() - torch.mean(targets.float())
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        pearson_correlation = 1 - cost  # 1 minus the Pearson correlation coefficient
        
        total_loss = self.weight_mae * mae + self.weight_pearson * pearson_correlation + self.weight_ce * ce
        if task == 'seg':
            return total_loss
        else:
            return ce


criterion = CombinedLoss(weight_pearson=0.33, weight_mae=0.33, weight_ce=0.33)

model = I2CNet(in_planes=16)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# min_val = -94.46878814697266
# max_val = 138.81414794921875
min_val = -95
max_val = 139

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # print(f'Original labels - Min: {labels.min().item()}, Max: {labels.max().item()}')
        labels = (labels - min_val) / (max_val - min_val)
        if width > 1:
            # print('labels prev',labels.shape)
            labels = labels.reshape(labels.shape[0], labels.shape[1], -1, window_size//width).mean(axis=2)
            # print('labels after',labels.shape)
        if acc is not None:
            labels = torch.floor(labels * (180 / acc))
            # labels[labels < 0] = 0  
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    scheduler.step()
    epoch_loss = running_loss/len(dataloader)
    loss_log.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

print("Finished Training")

logs = pd.DataFrame(loss_log, columns=['losses'])
logs.to_excel(config.save_path+'loss_logs.xlsx', index=False)
torch.save(model.state_dict(), config.save_path + f'model_{time.time()}.pth')
