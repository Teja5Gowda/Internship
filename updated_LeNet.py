import torch as th
import torch.nn as nn
#from lenet_network_prune import Network
from updated_network_prune import Network
from torchsummary import summary
import torch.nn.functional as F
#from updated_pruningmethod import PruningMethod

class LeNet(nn.Module, Network):

    def __init__(self,a_type, n_iterations, n_iterations_training):

        super(LeNet, self).__init__()

        self.a_type = a_type

        if a_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif a_type == 'tanh':
            self.activation = nn.Tanh()
        elif a_type == 'relu':
            self.activation = nn.ReLU()
        elif a_type == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif a_type == 'prelu':
            self.activation = nn.PRelu()
        elif a_type == 'elu':
            self.activation = nn.ELU()
        elif a_type == 'selu':
            self.activation = nn.SELU()
        elif a_type == 'gelu':
            self.activation = nn.Tanh()
        elif a_type == 'celu':
            self.activation = nn.CELU()
        elif a_type == 'softplus':
            self.activation = nn.Softplus()
        elif a_type == 'mish':
            self.activation = nn.Mish()
        else:
            print('Not implemented')
            raise
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Sequential(*([nn.Conv2d(1, 20, 5),self.activation]))

        #self.conv2 = nn.Sequential(
                #*([nn.Conv2d(20, 50, 5),self.activation]))

        #self.fc1 = nn.Sequential(
                #*([nn.Linear(20 * 12 * 12, 800),self.activation]))  # 5x5 image dimension

        #self.fc2 = nn.Sequential(
                #*([nn.Linear(800, 500),self.activation]))

        self.fc3 = nn.Linear(20 * 12 * 12, 10)

        for m in self.modules():
            self.weight_init(m)

        #self.pool_layer = nn.MaxPool2d(2, 2)
        #self.softmax = nn.Softmax(dim=1)

        self.sigmas = th.zeros((n_iterations))
        self.MI = th.zeros((n_iterations,2))

        self.sigmas_training = th.zeros((n_iterations_training))
        self.MI_training = th.zeros((n_iterations_training,2))

        
        self.test_accuracy=[]
        self.train_accuracy=[]

        self.train_loss=[]
        self.test_loss=[]


    def forward(self, x):
        #layer1 = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #layer2 = F.max_pool2d(F.relu(self.conv2(layer1)), 2)

        #layer1 = F.relu(self.conv1(x))
        layer1 = self.conv1(x)
        # print('layer1 -> ',layer1.shape)
        layer1_pool = F.max_pool2d(layer1, (2, 2))
        # print('layer1..',layer1_pool.shape)
        
        layer2_p = layer1_pool.view(-1, int(layer1_pool.nelement() / layer1_pool.shape[0]))

        # print('layer 2 pooled reshaped....',layer2_p.shape)

        layer5 = self.fc3(layer2_p)
        return layer5
#print(th.cuda.is_available())
device = th.device("cuda" if th.cuda.is_available() else "cpu") # PyTorch v0.4.0
model_tupry = LeNet('relu',100,200).to(device)
summary(model_tupry,(1,28,28))
#print(model_tupry)
