'''
Torch Model
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../Data/clean_data.csv')
data = data.to_numpy()
data = np.delete(data, 8, 1)
data = np.delete(data, 8, 1)
data = data.astype('float64')
np.random.shuffle(data)
class_label = data[:, -1] # for last column
class_label = class_label.astype('int64') 
data = data[:, :-1] # for all but last column
print(data.shape)
print(class_label.shape)

x_train = data[0:49000, :]
y_train = class_label[0:49000] 
x_val = data[49000:50000, :]
y_val = class_label[49000:50000] 
x_test = data[50000:, :]
y_test = class_label[50000:] 

print("Training Set Data  Shape: ", x_train.shape)
print("Training Set Label Shape: ", y_train.shape)
print("Validation Set Data  Shape: ", x_val.shape)
print("Validation Set Label Shape: ", y_val.shape)
print("Test Set Data  Shape: ", x_test.shape)
print("Test Set Label Shape: ", y_test.shape)

import torch
import torch.nn as nn
import torch.optim as optim
print(torch.cuda.get_device_name(0))
dtype = torch.float32
device = torch.device('cuda')

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_val = torch.from_numpy(x_val)
y_val = torch.from_numpy(y_val)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

print("Training Set Data  Shape: ", x_train.size())
print("Training Set Label Shape: ", y_train.size())
print("Validation Set Data  Shape: ", x_val.size())
print("Validation Set Label Shape: ", y_val.size())
print("Test Set Data  Shape: ", x_test.size())
print("Test Set Label Shape: ", y_test.size())

'''
Barebone API
'''

def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)
    
def check_accuracy(model_fn, params):
    print('Checking accuracy on the validation set')
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        x_val_gpu = x_train.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y_val_gpu = y_train.to(device=device, dtype=torch.int64)
        scores = model_fn(x_val_gpu, params)
        _, preds = scores.max(1)
        num_correct += (preds == y_val_gpu).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    
def train(model_fn, params, learning_rate):
    # Move the data to the proper device (GPU or CPU)
    x_train_gpu = x_train.to(device=device, dtype=dtype)
    y_train_gpu = y_train.to(device=device, dtype=torch.long)
    
    for i in range(10000):
        # Forward pass: compute scores and loss
        scores = model_fn(x_train_gpu, params)
        loss = F.cross_entropy(scores, y_train_gpu)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad

                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        print('loss = %.4f' % (loss.item()))
        check_accuracy(model_fn, params)
        print()
        
import torch.nn.functional as F  # useful stateless functions

def two_layer_fc(x, params):   
    w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7 = params
    x = F.relu(x.mm(w1) + b1)
    x = F.relu(x.mm(w2) + b2)
    x = F.relu(x.mm(w3) + b3)
    x = F.relu(x.mm(w4) + b4)
    x = F.relu(x.mm(w5) + b5)
    x = F.relu(x.mm(w6) + b6)
    x = x.mm(w7) + b7
    return x
    
hidden_layer_size_1 = 8
hidden_layer_size_2 = 7
hidden_layer_size_3 = 6
hidden_layer_size_4 = 5
hidden_layer_size_5 = 4
hidden_layer_size_6 = 3
learning_rate = 1e-3

w1 = random_weight((9, hidden_layer_size_1))
b1 = zero_weight(hidden_layer_size_1)

w2 = random_weight((hidden_layer_size_1, hidden_layer_size_2))
b2 = zero_weight(hidden_layer_size_2)

w3 = random_weight((hidden_layer_size_2, hidden_layer_size_3))
b3 = zero_weight(hidden_layer_size_3)

w4 = random_weight((hidden_layer_size_3, hidden_layer_size_4))
b4 = zero_weight(hidden_layer_size_4)

w5 = random_weight((hidden_layer_size_4, hidden_layer_size_5))
b5 = zero_weight(hidden_layer_size_5)

w6 = random_weight((hidden_layer_size_5, hidden_layer_size_6))
b6 = zero_weight(hidden_layer_size_6)

w7 = random_weight((hidden_layer_size_6, 2))
b7 = zero_weight(2)

train(two_layer_fc, [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7], learning_rate)

'''
nn.Module API
'''

def check_accuracy_module(model):
    print('Checking accuracy on training set')  
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        x_train_gpu = x_train.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y_train_gpu = y_train.to(device=device, dtype=torch.long)
        scores = model(x_train_gpu)
        _, preds = scores.max(1)
        num_correct += (preds == y_train_gpu).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    
def train_module(model, optimizer, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        model.train()  # put model to training mode
        x_train_gpu = x_train.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y_train_gpu = y_train.to(device=device, dtype=torch.long)

        scores = model(x_train_gpu)
        loss = F.cross_entropy(scores, y_train_gpu)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        loss.backward()

        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()

        if e % 100 == 0:
            print('Iteration %d, loss = %.4f' % (e, loss.item()))
            check_accuracy_module(model)
            print()
            
class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        #nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, x):
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores
        
hidden_layer_size = 8
learning_rate = 1e-3
model_fc = TwoLayerFC(9, hidden_layer_size, 2)
optimizer = optim.SGD(model_fc.parameters(), lr=learning_rate)
optimizer_1 = optim.SGD(model_fc.parameters(), lr=learning_rate/2)

train_module(model_fc, optimizer, 30000)
train_module(model_fc, optimizer_1, 10000)

b = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16, 17]])
torch.reshape(b, (2,1,3,3))

x_train_reshape = torch.reshape(x_train, (49000, 1, 3, 3))
x_val_reshape = torch.reshape(x_val, (1000, 1, 3, 3))
x_test_reshape = torch.reshape(x_test, (1598, 1, 3, 3))

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, 3, padding=1)
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)
        self.fc1 = nn.Linear(channel_2 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = flatten(x)
        scores = self.fc1(x)
        return scores
        
def train_module_conv(model, optimizer, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        model.train()  # put model to training mode
        x_train_gpu = x_train_reshape.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y_train_gpu = y_train.to(device=device, dtype=torch.long)

        scores = model(x_train_gpu)
        loss = F.cross_entropy(scores, y_train_gpu)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        loss.backward()

        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()

        if e % 100 == 0:
            print('Iteration %d, loss = %.4f' % (e, loss.item()))
            check_accuracy_module_conv(model)
            print()
            
def check_accuracy_module_conv(model):
    print('Checking accuracy on training set')  
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        x_train_gpu = x_train_reshape.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y_train_gpu = y_train.to(device=device, dtype=torch.long)
        scores = model(x_train_gpu)
        _, preds = scores.max(1)
        num_correct += (preds == y_train_gpu).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    
learning_rate = 3e-3
channel_1 = 32
channel_2 = 16

model_conv = ThreeLayerConvNet(1, channel_1, channel_2, 2)
optimizer = optim.SGD(model_conv.parameters(), lr=learning_rate)

train_module_conv(model_conv, optimizer, 10000)

print("training set")
tp = 0
fp = 0
fn = 0
num_correct = 0
num_samples = 0
model_conv.eval()  # set model to evaluation mode
with torch.no_grad():
    x_train_gpu = x_train_reshape.to(device=device, dtype=dtype)  # move to device, e.g. GPU
    y_train_gpu = y_train.to(device=device, dtype=torch.long)
    scores = model_conv(x_train_gpu)
    scores, preds = scores.max(1)
    preds_np = preds.cpu().numpy()
    y_train_gpu_np = y_train_gpu.cpu().numpy()
    num_correct += (preds == y_train_gpu).sum()
    num_samples += preds.size(0)
acc = float(num_correct) / num_samples
print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
for i in range(preds_np.shape[0]):
    if y_train_gpu_np[i] == 1:
        if preds_np[i] == 1:
            tp += 1
        else:
            fn += 1
    if y_train_gpu_np[i] == 0:
        if preds_np[i] == 1:
            fp += 1
            
print("tp: ", tp)
print("fp: ", fp)
print("fn: ", fn)
print("precision: ", tp/(tp+fp))
print("recall: ", tp/(tp+fn))

print("validation set")
tp = 0
fp = 0
fn = 0
num_correct = 0
num_samples = 0
model_conv.eval()  # set model to evaluation mode
with torch.no_grad():
    x_train_gpu = x_val_reshape.to(device=device, dtype=dtype)  # move to device, e.g. GPU
    y_train_gpu = y_val.to(device=device, dtype=torch.long)
    scores = model_conv(x_train_gpu)
    scores, preds = scores.max(1)
    preds_np = preds.cpu().numpy()
    y_train_gpu_np = y_train_gpu.cpu().numpy()
    num_correct += (preds == y_train_gpu).sum()
    num_samples += preds.size(0)
acc = float(num_correct) / num_samples
print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
for i in range(preds_np.shape[0]):
    if y_train_gpu_np[i] == 1:
        if preds_np[i] == 1:
            tp += 1
        else:
            fn += 1
    if y_train_gpu_np[i] == 0:
        if preds_np[i] == 1:
            fp += 1
            
print("tp: ", tp)
print("fp: ", fp)
print("fn: ", fn)
print("precision: ", tp/(tp+fp))
print("recall: ", tp/(tp+fn))

print("testing set")
tp = 0
fp = 0
fn = 0
num_correct = 0
num_samples = 0
model_conv.eval()  # set model to evaluation mode
with torch.no_grad():
    x_train_gpu = x_test_reshape.to(device=device, dtype=dtype)  # move to device, e.g. GPU
    y_train_gpu = y_test.to(device=device, dtype=torch.long)
    scores = model_conv(x_train_gpu)
    scores, preds = scores.max(1)
    preds_np = preds.cpu().numpy()
    y_train_gpu_np = y_train_gpu.cpu().numpy()
    num_correct += (preds == y_train_gpu).sum()
    num_samples += preds.size(0)
acc = float(num_correct) / num_samples
print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
for i in range(preds_np.shape[0]):
    if y_train_gpu_np[i] == 1:
        if preds_np[i] == 1:
            tp += 1
        else:
            fn += 1
    if y_train_gpu_np[i] == 0:
        if preds_np[i] == 1:
            fp += 1
            
print("tp: ", tp)
print("fp: ", fp)
print("fn: ", fn)
print("precision: ", tp/(tp+fp))
print("recall: ", tp/(tp+fn))