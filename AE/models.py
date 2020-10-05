from torch import nn

class SimpleAE(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(kwargs["input_shape"], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, kwargs["output_shape"])) 

        self.decoder = nn.Sequential(
            nn.Linear(kwargs["output_shape"], 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, 256), 
            nn.ReLU(),
            nn.Linear(256, 512), 
            nn.ReLU(), 
            nn.Linear(512, kwargs["input_shape"]),
            nn.Tanh())

    def forward(self, x):
        '''
        x: batch_features
        '''
        x = self.encoder(x)
        return self.decoder(x)

class ConvAE():
    '''
    https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
    '''

class VariationalAE():
    '''
    https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/Variational_autoencoder.py
    '''
