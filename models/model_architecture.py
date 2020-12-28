import torch.nn as nn
import torch

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        #using sequential helps bind multiple operations together
        self.layer1 = nn.Sequential(
            #in_channel = 1
            #out_channel = 16
            #padding = (kernel_size - 1) / 2 = 2
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #after layer 1 will be of shape [100, 16, 14, 14]
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #after layer 2 will be of shape [100, 32, 7, 7]
        self.fc = nn.Linear(32*7*7, num_classes)
        self.drop_out = nn.Dropout(p=0.2)  #zeroed 0.2% data
        #after fc will be of shape [100, 10]
        
    def forward(self, x):
        #x shape: [batch, in_channel, img_width, img_height]
        #[100, 1, 28, 28]
        out = self.layer1(x)
        out = self.drop_out(out)
        #after layer 1: shape: [100, 16, 14, 14]
        out = self.layer2(out)
        out = self.drop_out(out)
        #after layer 2: shape: [100, 32, 7, 7]
        out = out.reshape(out.size(0), -1)
        #after squeezing: shape: [100, 1568]
        #we squeeze so that it can be inputted into the fc layer
        out = self.fc(out)
        #after fc layer: shape: [100, 10]
        return out

class Model:
    def __init__(self):
        self.model = ConvNet() 
        self.device = torch.device('cpu')
        # load model file 
        
        try :
            path = 'models/cnn_digits.ckpt'
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        except : 
            path = 'cnn_digits.ckpt'
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        
    def predict(self, data):
        data = torch.from_numpy(data).type(torch.FloatTensor)
        outputs = self.model(data)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()
            