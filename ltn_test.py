import torchvision.transforms as transforms
import torch
import ltn
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from EyeDataset import EyeDataset
from CLMCDataset import CLMCDataset
from CICIDSDataset import CICIDSDataset
import os
import numpy as np
from PIL import Image
import time

n_epochs = 8
learning_rate = 0.001
momentum = 0.5
log_interval = 100
batch_size = 64
num_class = 1
IMG_SIZE = 128
n_features = 4

# root = '../dataset/eye/ODIR-5K/ODIR-5K/'
# train_file = 'train.csv'
# test_file = 'test.csv'
root = '../dataset/n/'
train_file = 'CICIDS_train.csv'
test_file = 'CICIDS_test.csv'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
transform_data = transforms.Compose(
    [transforms.ToTensor()])

# dataset = EyeDataset(train_file, root+'/Training Images', transform_img=transform, transform_data=transform_data)
# num_train = int(len(dataset)*0.8)
# num_test = len(dataset)-num_train
# train_data, test_data, _ = torch.utils.data.random_split(dataset,[100, 20, len(dataset)-120])
train_data = CICIDSDataset(train_file, root, transform_img=transform, transform_data=transform_data)
test_data = CICIDSDataset(test_file, root, transform_img=transform, transform_data=transform_data)
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

if not os.path.exists('./results'):
    os.mkdir('./results')

labels = ltn.Variable("d", torch.tensor(range(num_class)))

# l = dict()
# for d in range(num_class):
#     l[d] = (ltn.Constant(torch.tensor(d)))
#

# Predicate D
class DecisionTreePredicate(nn.Module):
    def __init__(self):
        super(DecisionTreePredicate, self).__init__()
        self.decision_tree = RandomForestClassifier(n_estimators=10)

        self.name = 'DT'

    def forward(self, input_data, target_labels):
        input_data_numpy = input_data.cpu().numpy()

        self.decision_tree.fit(input_data_numpy, target_labels)

        output = self.decision_tree.predict_proba(input_data_numpy)
        output_tensor = torch.tensor(output[:,1], dtype=torch.float).to(input_data.device)
        return output_tensor

    def predict(self, input_data):
        return self.decision_tree.predict(input_data)


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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        out = self.softmax(logits)
        return out

f_mlp = CNN()
c_mlp = DNN()
model = ltn.Predicate(LogitsToPredicate(f_mlp))
modelDT = ltn.Predicate(LogitsToPredicate(c_mlp))
# we define the connectives, quantifiers, and the SatAgg
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=4))

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

batch_counter = 0

train_losses = []
train_counter = []
test_losses = []

def equal(x,z,y):
    a = torch.eq(x, y)
    b = torch.eq(z,y)
    return torch.mul(a, b)

x = ltn.Variable("x", torch.tensor(range(1)))
z = ltn.Variable("z", torch.tensor(range(1)))

def train_ltn(epoch, p):
    global batch_counter
    total_correct_CNN = 0
    total_correct_DNN = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_data_loader):
        optimizer.zero_grad()

        images_x = ltn.Variable("x", data['CT'])
        clinical_x = ltn.Variable("z", data['Clinical'])
        labels_y = ltn.Variable("y", target)

        # sat_agg = And(
        #     Forall(
        #         ltn.diag(images_x, labels_y),
        #         model(images_x, labels_y)),
        #     Forall(
        #         ltn.diag(clinical_x, labels_y),
        #         modelDT(clinical_x, labels_y))).value
        sat_agg = Forall(
            ltn.diag(images_x, clinical_x, labels_y),
            Exists([x, z],
                   And(model(images_x, x), modelDT(clinical_x, z)),
                   cond_vars = [x, z, labels_y],
                   cond_fn = lambda x, z, y: equal(x.value, z.value, y.value),
                   p = p
                   )).value

        train_loss = 1. - sat_agg

        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            outCNN = f_mlp(data['CT'])
            pred = outCNN.data.max(1, keepdim=True)[1]
            total_correct_CNN += pred.eq(target.data.view_as(pred)).sum()
            outDNN = c_mlp(data['Clinical'])
            pred = outDNN.data.max(1, keepdim=True)[1]
            total_correct_DNN += pred.eq(target.data.view_as(pred)).sum()
            total_samples += target.size(0)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_data_loader.dataset),
                       100. * batch_idx / len(train_data_loader), train_loss.item()))
            train_losses.append(train_loss.item())
            batch_counter += log_interval
            train_counter.append(batch_counter)
            torch.save(model.state_dict(), './results/model_ltn.pth')

    accuracy_DNN = 100 * total_correct_DNN / total_samples
    accuracy_CNN = 100 * total_correct_CNN / total_samples
    print(f'Accuracy CNN on train set: {accuracy_CNN:.2f}% \nAccuracy DNN on train set: {accuracy_DNN:.2f}%')


test_counter = [i * len(train_data_loader.dataset) / batch_size for i in range(n_epochs + 1)]


def test_ltn(loader, set):
    model.eval()
    correct_dt = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            ct, clinic, labels = data['CT'].to(ltn.device), data['Clinical'].to(ltn.device), target.to(ltn.device)
            output = f_mlp(ct)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
            out_dt = c_mlp(clinic)
            pred_dt = out_dt.data.max(1, keepdim=True)[1]
            correct_dt += pred.eq(labels.data.view_as(pred_dt)).sum()

    print('{} set:\n'
          '\tAvg. Accuracy CNN: {}/{} ({:.0f}%)\n'
          '\tAvg. Accuracy Decision Tree: {}/{} ({:.0f}%)'.format(set,
                                                                  correct,
                                                                  len(loader.dataset),
                                                                  100. * correct / len(
                                                                      loader.dataset),
                                                                  correct_dt,
                                                                  len(loader.dataset),
                                                                  100. * correct_dt / len(
                                                                      loader.dataset)))


for epoch in range(1, n_epochs + 1):
    start = time.time()
    if epoch in range(0, 10):
        p = 1
    if epoch in range(10, 20):
        p = 2
    if epoch in range(20, 30):
        p = 4
    if epoch in range(30, 40):
        p = 6
    train_ltn(epoch, p)
    end_train = time.time()
    print(f"Time training: {end_train - start} sec.")
    test_ltn(test_data_loader, "Test")
    end_test = time.time()
    print(f"Time test: {end_test - end_train} sec.\n")


# # Converti l'immagine in un array numpy
        # fake_clinical_array = torch.Tensor(np.random.rand(batch_size, n_features))
        # unique_values = np.arange(8)
        # if batch_size < 8:
        #     raise ValueError(
        #         "Il batch_size deve essere almeno 8 per garantire che ogni valore sia presente almeno una volta.")
        #
        # repetitions = batch_size // 8
        # remainder = batch_size % 8
        #
        # repeated_values = np.tile(unique_values, repetitions)
        # extra_values = np.random.choice(unique_values, remainder, replace=False)
        #
        # y_batch = np.concatenate((repeated_values, extra_values))
        # # y_train_artificial = torch.Tensor(np.eye(num_class)[y_batch]) # Classi da 0 a 7
        #
        # # Addestrare il modello sul set di addestramento artificiale
        # self.decision_tree.fit(fake_clinical_array, y_batch)