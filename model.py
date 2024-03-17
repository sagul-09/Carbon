#perfect code -1
import torch
import torch.nn as nn
import torch.nn.functional as F
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, hidden_size)  # Added another hidden layer
        self.l4 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)  # Added batch normalization
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)  # Added batch normalization
    
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.batchnorm1(out)  # Apply batch normalization
        out = self.dropout(out)
        out = self.relu(self.l2(out))
        out = self.batchnorm2(out)  # Apply batch normalization
        out = self.dropout(out)
        out = self.relu(self.l3(out))  # Added another layer in the forward pass
        out = self.l4(out)
        return out
hidden_size = 128

