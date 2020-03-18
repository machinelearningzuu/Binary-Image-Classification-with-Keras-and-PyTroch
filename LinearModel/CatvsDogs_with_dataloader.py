from dataloader import load_data
import torch
from torch import nn as nn
from torch import optim as optim
import numpy as np
from variables import batch_size, input_size, output, keep_prob, learning_rate, num_epochs, verbose, dense1, dense2, dense3
import matplotlib.pyplot as plt
from time import time

class CatVsDogsClassifier(object):
    def __init__(self, is_model_heavy=True, cuda_compatible=True):
        torch.cuda.empty_cache()
        train_loader, test_loader = load_data()
        self.train_loader= train_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda:0') if cuda_compatible and is_model_heavy and torch.cuda.is_available() else torch.device('cpu')
        print("Running on {}".format(self.device))

    def Classifier(self):
        self.model = nn.Sequential(
                    nn.Linear(input_size, dense1),
                    nn.ReLU(),
                    nn.Linear(dense1, dense2),
                    nn.ReLU(),
                    nn.Linear(dense2, dense3),
                    nn.ReLU(),
                    nn.Dropout(keep_prob),
                    nn.Linear(dense3, output),
                    nn.Sigmoid()
                    ).to(self.device)

    def train_model(self):
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        train_loss = []
        test_loss = []
        t0 = time()

        n_corrcted_train = 0
        n_total_train = 0
        train_epoch_loss = 0

        steps = 0

        for epoch in range(num_epochs):
            for inputs, labels in self.train_loader:
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()

                self.model.zero_grad()
                flattened_input =  inputs.view(inputs.shape[0],-1)
                output_tensor = self.model.forward(flattened_input).squeeze()
                P = output_tensor.round()
                loss = loss_function(output_tensor, labels)

                train_epoch_loss += loss.item()
                n_corrcted_train += (P == labels).float().sum()
                n_total_train += len(P)

                loss.backward()
                optimizer.step()

                n_corrcted_test = 0
                n_total_test = 0
                test_epoch_loss = 0
                if steps % verbose == 0:
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.test_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                            flattened_input =  inputs.view(inputs.shape[0],-1)
                            output_tensor = self.model.forward(flattened_input).squeeze()
                            P = output_tensor.round()
                            batch_loss = loss_function(output_tensor, labels)

                            test_epoch_loss += batch_loss.item()
                            n_corrcted_test += (P == labels).float().sum()
                            n_total_test += len(P)

                    test_epoch_loss = round(test_epoch_loss,3)
                    train_epoch_loss = round(train_epoch_loss,3)
                    test_loss.append(test_epoch_loss)
                    train_loss.append(train_epoch_loss)
                    train_accuracy = round((n_corrcted_train / n_total_train).item(),3)
                    test_accuracy  = round((n_corrcted_test / n_total_test).item(),3)

                    iteration = steps // verbose
                    print("Epoch : {} iteration : {} , train_loss : {} train_accuracy : {} val_loss : {} val_accuracy : {}".format(epoch,iteration,train_epoch_loss,train_accuracy,test_epoch_loss,test_accuracy))
                    train_epoch_loss = 0
                    n_corrcted_train = 0
                    n_total_train = 0

                    self.model.train()

        t1 = time()
        print("running time with {} : {} ".format(self.device,t1-t0))
        plt.plot(np.array(train_loss),label="train_loss")
        plt.plot(np.array(train_loss),label="test_loss")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    model = CatVsDogsClassifier()
    model.Classifier()
    model.train_model()

