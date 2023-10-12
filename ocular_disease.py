from time import time

import torchvision.transforms as transforms
import torch
import ltn
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from EyeDataset import EyeDataset
import os
import numpy as np
from sklearn.metrics import accuracy_score
import torchvision

n_epochs = 8
learning_rate = 0.001
momentum = 0.5
log_interval = 100
batch_size = 32
num_class = 2
IMG_SIZE = 128
n_features = 4

root = '../dataset/eye/ODIR-5K/ODIR-5K/'
train_file = 'train.csv'
test_file = 'test.csv'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_data = transforms.Compose(
    [transforms.ToTensor()])

dataset = EyeDataset(train_file, root+'/Training Images', transform_img=transform, transform_data=transform_data)
num_train = int(len(dataset)*0.8)
num_test = len(dataset)-num_train
train_data, test_data = torch.utils.data.random_split(dataset,[num_train, num_test])
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

if not os.path.exists('./results'):
    os.mkdir('./results')

# Predicate clinical
class DNN(torch.nn.Module):
    # the convolutional neural network is the same as in the standard CNN example
    def __init__(self):
        super().__init__()
        self.name = 'Clinical'
        self.fc1 = nn.Linear(4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Predicate C
class CNN(torch.nn.Module):
    # the convolutional neural network is the same as in the standard CNN example
    def __init__(self):
        super().__init__()
        self.name = 'CNN'
        self.base_model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        for param in self.base_model.parameters():
            param.requires_grad = False
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        x = self.base_model(x)
        return x

class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label l. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class l.
    """

    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, l):
        logits = self.logits_model(x)
        probs = self.softmax(logits)
        out = torch.sum(probs * l, dim=1)
        return out


l_A = ltn.Constant(torch.tensor([1, 0]))
l_B = ltn.Constant(torch.tensor([0, 1]))

cnn = CNN()
print(cnn)
c_mlp = DNN()
model = ltn.Predicate(LogitsToPredicate(cnn))
model_clinical = ltn.Predicate(LogitsToPredicate(c_mlp))
# we define the connectives, quantifiers, and the SatAgg
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=4))


batch_counter = 0

train_losses = []
train_counter = []
test_losses = []

def compute_sat_level(loader):
    mean_sat = 0
    for data, labels in loader:
        x_A_sx = ltn.Variable("x_A_sx", data['SX'][labels == 0])
        x_A_dx = ltn.Variable("x_A_dx", data['DX'][labels == 0])
        x_B_sx = ltn.Variable("x_B_sx", data['SX'][labels == 1])
        x_B_dx = ltn.Variable("x_B_dx", data['DX'][labels == 1])
        z_A = ltn.Variable("z_A", data['Clinical'][labels == 0])
        z_B = ltn.Variable("z_B", data['Clinical'][labels == 1])
        mean_sat += SatAgg(
            Or(
                Forall(x_A_sx, model(x_A_sx, l_A)),
                Forall(x_A_dx, model(x_A_dx, l_A)),
            ),
            Or(
                Forall(x_B_sx, model(x_B_sx, l_B)),
                Forall(x_B_dx, model(x_B_dx, l_B)),
            ),
            Forall(z_A, model_clinical(z_A, l_A)),
            Forall(z_B, model_clinical(z_B, l_B)),
        )
    mean_sat /= len(loader)
    return mean_sat

def compute_accuracy(loader, mod, name):
    mean_accuracy = 0.0
    if name == 'Clinical':
        for data, labels in loader:
            predictions = torch.nn.Softmax(dim=1)(mod(data[name]))
            predictions = np.argmax(predictions.detach().numpy(), axis=1)
            mean_accuracy += accuracy_score(labels, predictions)
    else:
        for data, labels in loader:
            predictions_sx = torch.nn.Softmax(dim=1)(mod(data['SX']))
            predictions_dx = torch.nn.Softmax(dim=1)(mod(data['DX']))
            p_sx = np.argmax(predictions_sx.detach().numpy(), axis=1)
            p_dx = np.argmax(predictions_dx.detach().numpy(), axis=1)
            predictions = np.round((p_dx + p_sx)/2)
            mean_accuracy += accuracy_score(labels, predictions)

    return mean_accuracy / len(loader)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_clinical = torch.optim.Adam(model_clinical.parameters(), lr=0.001)

for epoch in range(10):
    train_loss = 0.0
    start = time()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        optimizer_clinical.zero_grad()
        # we ground the variables with current batch data
        x_A_sx = ltn.Variable("x_A_sx", data['SX'][labels == 0])
        x_A_dx = ltn.Variable("x_A_dx", data['DX'][labels == 0])
        x_B_sx = ltn.Variable("x_B_sx", data['SX'][labels == 1])
        x_B_dx = ltn.Variable("x_B_dx", data['DX'][labels == 1])
        z_A = ltn.Variable("z_A", data['Clinical'][labels == 0])
        z_B = ltn.Variable("z_B", data['Clinical'][labels == 1])
        sat_agg = SatAgg(
            Or(
                Forall(x_A_sx, model(x_A_sx, l_A)),
                Forall(x_A_dx, model(x_A_dx, l_A)),
            ),
            Or(
                Forall(x_B_sx, model(x_B_sx, l_B)),
                Forall(x_B_dx, model(x_B_dx, l_B)),
            ),
            Forall(z_A, model_clinical(z_A, l_A)),
            Forall(z_B, model_clinical(z_B, l_B)),
        )
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        optimizer_clinical.step()
        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)
    print("epoch %d| loss %.4f | Train Sat %.3f" % (epoch + 1,
                                                    train_loss,
                                                    1-train_loss))
    end = time()
    print("time per train one epoch: %d" % (end - start))
    # we print metrics every 20 epochs of training
    if epoch % 5 == 0:
        print("\t model CNN | Train Acc %.3f | Test Acc %.3f"
              %(compute_accuracy(train_loader, cnn, 'CT'),
                compute_accuracy(test_loader, cnn, 'CT')))
        print("\t model Clinical | Train Acc %.3f | Test Acc %.3f"
              %(compute_accuracy(train_loader, c_mlp, 'Clinical'),
                compute_accuracy(test_loader, c_mlp, 'Clinical')))
    end = time()
    print("time per epoch: %d"%(end-start))