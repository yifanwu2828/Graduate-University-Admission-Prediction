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

'''
Separate data into training data and testing data
'''

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

# Import more utilies and the layers you have implemented
from MLP.sequential import Sequential
from MLP.linear import Linear
from MLP.relu import ReLU
from MLP.softmax import Softmax
from MLP.loss_func import CrossEntropyLoss
from MLP.optimizer import SGD
from MLP.dataset import DataLoader
from MLP.trainer import Trainer

# test-case 1:
input_size = 9
hidden_size = 5 # Hidden layer size (Hyper-parameter)
num_classes = 2 # Output

def init_model():
    l1 = Linear(input_size, hidden_size)
    l2 = Linear(hidden_size, num_classes)
    
    r1 = ReLU()
    softmax = Softmax()
    return Sequential([l1, r1, l2, softmax])

# Initialize the dataset with the dataloader class
dataset = DataLoader(x_train, y_train, x_val, y_val, x_test, y_test)
net_1 = init_model()
optim = SGD(net_1, lr=0.0015, weight_decay=0.000)
loss_func = CrossEntropyLoss()
epoch = 6000
batch_size = 1225

#Initialize the trainer class by passing the above modules
trainer = Trainer(dataset, optim, net_1, loss_func, epoch, batch_size, validate_interval=3)

train_error_1, validation_accuracy_1 = trainer.train()

# test-case 1:
from MLP.evaluation import get_classification_accuracy
out_train = net_1.predict(x_train)
acc = get_classification_accuracy(out_train, y_train)
print("Training acc: ",acc)
out_val = net_1.predict(x_val)
acc = get_classification_accuracy(out_val, y_val)
print("Validation acc: ",acc)
test_acc = (net_1.predict(x_test) == y_test).mean()
print('Test accuracy: ', test_acc)

# Plot the training loss function and validation accuracies
plt.subplot(2, 1, 1)
plt.plot(train_error_1)
plt.title('Training Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
print()
plt.subplot(2, 1, 2)
#plt.plot(stats['train_acc_history'], label='train')
plt.plot(validation_accuracy_1, label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# test-case 2:
input_size = 9
hidden_size = 5 # Hidden layer size (Hyper-parameter)
num_classes = 2 # Output

def init_model():
    l1 = Linear(input_size, hidden_size)
    l2 = Linear(hidden_size, num_classes)
    
    r1 = ReLU()
    softmax = Softmax()
    return Sequential([l1, r1, l2, softmax])

# Initialize the dataset with the dataloader class
dataset = DataLoader(x_train, y_train, x_val, y_val, x_test, y_test)
net_2 = init_model()
net_2._modules[0].w = np.copy(net_1._modules[0].w)
net_2._modules[0].b = np.copy(net_1._modules[0].b)
net_2._modules[2].w = np.copy(net_1._modules[2].w)
net_2._modules[2].b = np.copy(net_1._modules[2].b)
optim = SGD(net_2, lr=0.00005, weight_decay=0.000)
loss_func = CrossEntropyLoss()
epoch = 6000
batch_size = 49000

#Initialize the trainer class by passing the above modules
trainer = Trainer(dataset, optim, net_2, loss_func, epoch, batch_size, validate_interval=3)

train_error_2, validation_accuracy_2 = trainer.train()

# test-case 2:
from MLP.evaluation import get_classification_accuracy
out_train = net_2.predict(x_train)
acc = get_classification_accuracy(out_train, y_train)
print("Training acc: ",acc)
out_val = net_2.predict(x_val)
acc = get_classification_accuracy(out_val, y_val)
print("Validation acc: ",acc)
test_acc = (net_2.predict(x_test) == y_test).mean()
print('Test accuracy: ', test_acc)

# Plot the training loss function and validation accuracies
plt.subplot(2, 1, 1)
plt.plot(train_error_2)
plt.title('Training Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
#plt.plot(stats['train_acc_history'], label='train')
plt.plot(validation_accuracy_2, label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.tight_layout()
plt.show()

print("trainins set")
out_train = net_2.predict(x_train)
tp = 0
fp = 0
fn = 0
num_correct = 0
num_samples = out_train.shape[0]
for i in range(out_train.shape[0]):
    if out_train[i] == y_train[i]:
        num_correct += 1
    if y_train[i] == 1:
        if out_train[i] == 1:
            tp += 1
        else:
            fn += 1
    if y_train[i] == 0:
        if out_train[i] == 1:
            fp += 1
acc = float(num_correct) / num_samples
print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
print("tp: ", tp)
print("fp: ", fp)
print("fn: ", fn)
print("precision: ", tp/(tp+fp))
print("recall: ", tp/(tp+fn))

print("validation set")
out_val = net_2.predict(x_val)
tp = 0
fp = 0
fn = 0
num_correct = 0
num_samples = out_val.shape[0]
for i in range(out_val.shape[0]):
    if out_val[i] == y_val[i]:
        num_correct += 1
    if y_val[i] == 1:
        if out_val[i] == 1:
            tp += 1
        else:
            fn += 1
    if y_val[i] == 0:
        if out_val[i] == 1:
            fp += 1
acc = float(num_correct) / num_samples
print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
print("tp: ", tp)
print("fp: ", fp)
print("fn: ", fn)
print("precision: ", tp/(tp+fp))
print("recall: ", tp/(tp+fn))

print("testing set")
out_test = net_2.predict(x_test)
tp = 0
fp = 0
fn = 0
num_correct = 0
num_samples = out_test.shape[0]
for i in range(out_test.shape[0]):
    if out_test[i] == y_test[i]:
        num_correct += 1
    if y_test[i] == 1:
        if out_test[i] == 1:
            tp += 1
        else:
            fn += 1
    if y_test[i] == 0:
        if out_test[i] == 1:
            fp += 1
acc = float(num_correct) / num_samples
print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
print("tp: ", tp)
print("fp: ", fp)
print("fn: ", fn)
print("precision: ", tp/(tp+fp))
print("recall: ", tp/(tp+fn))