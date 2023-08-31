import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField


df = pd.read_csv('Datasets/EEGEyeState/EEG Eye State.csv')
df_distance_learning = pd.read_csv('Datasets/DistanceLearning/EEG_data.csv')
df_epileptic_seziure = pd.read_csv('Datasets/EpilepticSeizure/Epileptic Seizure Recognition.csv')
df_epileptic_seziure = df_epileptic_seziure.drop(['Unnamed'], axis=1)
df_epileptic_seziure['y'] = df_epileptic_seziure['y'].replace([2,3,4,5], 0)
df_distance_learning = df_distance_learning.drop(df_distance_learning.index[9595:], axis=0)
df_distance_learning = df_distance_learning.drop(['video_id', 'subject_id'], axis=1)

# Eye State Split
train, test = train_test_split(df, test_size=0.2, random_state=0)
# Distance Learning Split
train_dl, test_dl = train_test_split(df_distance_learning, test_size=0.2, random_state=0)
# Epileptic Seizure Split
train_es, test_es = train_test_split(df_epileptic_seziure, test_size=0.2, random_state=0)


def convert_gaf(train, test):
    gaf = GramianAngularField()
    trains = gaf.fit_transform(train.iloc[:, 0:-1])
    tests = gaf.transform(test.iloc[:, 0:-1])
    return trains, tests


def convert_mtf(train, test):
    mtf = MarkovTransitionField()
    trains_mtf = mtf.fit_transform(train.iloc[:, 0:-1])
    tests_mtf = mtf.transform(test.iloc[:, 0:-1])
    return trains_mtf, tests_mtf


distance_gaf_train, distance_gaf_test = convert_gaf(train_dl, test_dl)
distance_mtf_train, distance_mtf_test = convert_mtf(train_dl, test_dl)

epileptic_gaf_train, epileptic_gaf_test = convert_gaf(train_es, test_es)
epileptic_mtf_train, epileptic_mtf_test = convert_mtf(train_es, test_es)

eye_gaf_train, eye_gaf_test = convert_gaf(train, test)
eye_mtf_train, eye_mtf_test = convert_mtf(train, test)


out_eye = train.eyeDetection.to_numpy().reshape(-1,1)
out_eye_test = test.eyeDetection.to_numpy().reshape(-1,1)

out_distance = train_dl.subject_understood.to_numpy().reshape(-1,1)
out_distance_test = test_dl.subject_understood.to_numpy().reshape(-1,1)

out_epileptic = train_es.y.to_numpy().reshape(-1,1)
out_epileptic_test = test_es.y.to_numpy().reshape(-1,1)


def to_data_loader(trains, tests, out_train, out_test):
    train_data = []
    for i in range(len(trains)):
        train_data.append([trains[i], out_train[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=1)

    test_data = []
    for i in range(len(tests)):
        test_data.append([tests[i], out_test[i]])

    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=1)

    return trainloader, testloader


distance_trainloader_gaf, distance_testloader_gaf = to_data_loader(distance_gaf_train, distance_gaf_test, out_distance,
                                                                    out_distance_test)
distance_trainloader_mtf, distance_testloader_mtf = to_data_loader(distance_mtf_train, distance_mtf_test, out_distance,
                                                                    out_distance_test)
eye_trainloader_gaf, eye_testloader_gaf = to_data_loader(eye_gaf_train, eye_gaf_test, out_eye,
                                                                    out_eye_test)
eye_trainloader_mtf, eye_testloader_mtf = to_data_loader(eye_mtf_train, eye_mtf_test, out_eye,
                                                                    out_eye_test)
pileptic_trainloader_gaf, epileptic_testloader_gaf = to_data_loader(epileptic_gaf_train, epileptic_gaf_test, out_epileptic,
                                                                    out_epileptic_test)
epileptic_trainloader_mtf, epileptic_testloader_mtf = to_data_loader(epileptic_mtf_train, epileptic_mtf_test, out_epileptic,
                                                                    out_epileptic_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
epochs = 25
batch_size = 1
learning_rate = 0.01


class ConvNet(nn.Module):
    def __init__(self, num_of_features):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.num_of_features = int((num_of_features - 3 + 1) / 2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # 2,2
        self.num_of_features = int((self.num_of_features - 5 + 1) / 2)

        self.fc1 = nn.Linear(16 * self.num_of_features * self.num_of_features, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 16 * self.num_of_features * self.num_of_features)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


n_correct = 0
n_samples = 0
# Different models created for three different datasets
model = ConvNet(eye_mtf_train.shape[1]).to(device)  # Use this for eye state dataset
# model = ConvNet(distance_mtf_train.shape[1]).to(device) # Use this for epileptic seziure dataset
# model = ConvNet(epileptic_mtf_train.shape[1]).to(device) # Use this for distance learning dataset

model.double()
criterion = nn.BCEWithLogitsLoss()
criterion.double()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps = eye_mtf_train.shape[0]  # Use this for eye state dataset
# n_total_steps = distance_mtf_train.shape[0] #  Use this for distance learning dataset
# n_total_steps = epileptic_mtf_train.shape[0] # Use this for epileptic seziure dataset

losses_training = []
losses_test = []
loss_train = 0
loss_test = 0
for epc in range(epochs):
    loss_train = 0
    # eye_trainloader_mtf, distance_trainloader_mtf, epileptic_trainloader_mtf
    for i, (images, labels) in enumerate(eye_trainloader_mtf):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs.double(), labels.double())

        predicted = [1 if torch.sigmoid(outputs) >= 0.5 else 0]
        predicted = np.array(predicted)
        n_samples += labels.size(0)
        if predicted[0] == labels[0]:
            n_correct += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the training set: {acc} %')
    losses_training.append(loss_train / n_total_steps)
print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
# Testing the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    # eye_testloader_mtf distance_testloader_mtf, epileptic_testloader_mtf
    for i, (images, labels) in enumerate(epileptic_testloader_mtf):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        predicted = [1 if torch.sigmoid(outputs) >= 0.5 else 0]
        predicted = np.array(predicted)
        n_samples += labels.size(0)
        if predicted[0] == labels[0]:
            n_correct += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the test set: {acc} %')

plt.plot(losses_training)
plt.show()
