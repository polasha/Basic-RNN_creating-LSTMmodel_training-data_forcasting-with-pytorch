#Md.Surat-E-Mostafa

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = torch.linspace(0,799,800)
#print(x)
y = torch.sin (x*2*3.1416/40)
print(y)

plt.figure(figsize=(12,4))
plt.xlim(-10,801)
plt.grid(True)
plt.plot(y.numpy())
plt.title("Simple SineWave ")
plt.show()


test_size = 40
train_set = y[: -test_size]
test_set = y[-test_size:]
plt.figure(figsize=(12,4))
plt.xlim(-10,801)
plt.grid(True)
plt.plot(train_set.numpy())
plt.title("Slected training part of SineWave ")
plt.show()



def input_data(seq,ws):
    out = []
    L = len(seq)

    for i in range(L-ws):
        window = seq[i : i+ws]
        label = seq[i+ws : i+ws+1]
        out.append((window,label))

    return out



window_size = 40
train_data = input_data(train_set, window_size)
print(train_data)
print(len(train_data))


#LSTM model

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50,out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size,out_size)
        self.hidden = ( torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred= self.linear(lstm_out.view(len(seq), -1))

        return pred[-1]


torch.manual_seed(42)
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


print(model)


for p in model.parameters():
    print(p.numel())



#how to training and forcasting


epochs = 10
future =40

for i in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1,1, model.hidden_size), torch.zeros(1,1, model.hidden_size))
        y_pred = model(seq)

        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

print( f" Epoch {i} Loss:  {loss.item()}")


preds = train_set [-window_size:].tolist()

for f in range(future):
    seq = torch.FloatTensor(preds [-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1,1, model.hidden_size), torch.zeros(1,1, model.hidden_size))

        preds.append(model(seq).item())

loss = criterion(torch.tensor(preds[-window_size:]), y[760:])
print( f" performance on the test range : {loss}")


plt.figure(figsize=(12,4))
plt.xlim(700,801)
plt.grid(True)
plt.plot(y.numpy())
plt.plot(range(760,800), preds[-window_size:])
plt.title("Forcasting based on the training data and compare with test data")
plt.show()


# for retrain model for performance improvement

epochs = 15
window_size = 40
future = 40
all_data = input_data(y, window_size)
print(len(all_data))

import time
start_time = time.time()

for i in range(epochs):
    for seq, y_train in all_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1,1, model.hidden_size), torch.zeros(1,1, model.hidden_size))
        y_pred = model(seq)

        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

print( f" Epoch {i} Loss:  {loss.item()}")

total_time = time.time() - start_time
print(total_time/60)


#forecast into unknown future

preds = y[-window_size:].tolist()

for f in range(future):
    seq = torch.FloatTensor(preds [-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1,1, model.hidden_size), torch.zeros(1,1, model.hidden_size))

        preds.append(model(seq).item())

plt.figure(figsize=(12,4))
plt.xlim(0,841)
plt.grid(True)
plt.plot(y.numpy())

# plot forecast

plt.plot(range(800,800+future), preds[-window_size:])
plt.title("Forcasting the model with unknown data")
plt.show()







