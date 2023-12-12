import torch
import torch.nn as nn

class VAE_MLP_encoder(nn.Module):
    """
    MLP encoder class for the VAE model.
    """
    def __init__(self,params):
        """
        Required input parameters:
        - seq_len: (Int) Sequence length of sequence alignment
        - alphabet_size: (Int) Alphabet size of sequence alignment (will be driven by the data helper object)
        - hidden_layers_sizes: (List) List of sizes of DNN linear layers
        - z_dim: (Int) Size of latent space
        - convolve_input: (Bool) Whether to perform 1d convolution on input (kernel size 1, stide 1)
        - convolution_depth: (Int) Size of the 1D-convolution on input
        - nonlinear_activation: (Str) Type of non-linear activation to apply on each hidden layer
        - dropout_proba: (Float) Dropout probability applied on all hidden layers. If 0.0 then no dropout applied
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = params['seq_len']
        self.alphabet_size = params['alphabet_size']
        self.hidden_layers_sizes = params['hidden_layers_sizes']
        self.z_dim = params['z_dim']
        self.convolve_input = params['convolve_input']
        self.convolution_depth = params['convolution_input_depth']
        self.dropout_proba = params['dropout_proba']

        self.mu_bias_init = 0.1
        self.log_var_bias_init = -10.0

        #Convolving input with kernels of size 1 to capture potential similarities across amino acids when encoding sequences
        if self.convolve_input:
            self.input_convolution = nn.Conv1d(in_channels=self.alphabet_size,out_channels=self.convolution_depth,kernel_size=1,stride=1,bias=False)
            self.channel_size = self.convolution_depth
        else:
            self.channel_size = self.alphabet_size

        self.hidden_layers=torch.nn.ModuleDict()
        for layer_index in range(len(self.hidden_layers_sizes)):
            if layer_index==0:
                self.hidden_layers[str(layer_index)] = nn.Linear((self.channel_size*self.seq_len),self.hidden_layers_sizes[layer_index])
                nn.init.constant_(self.hidden_layers[str(layer_index)].bias, self.mu_bias_init)
            else:
                self.hidden_layers[str(layer_index)] = nn.Linear(self.hidden_layers_sizes[layer_index-1],self.hidden_layers_sizes[layer_index])
                nn.init.constant_(self.hidden_layers[str(layer_index)].bias, self.mu_bias_init)
        
        self.fc_mean = nn.Linear(self.hidden_layers_sizes[-1],self.z_dim)
        nn.init.constant_(self.fc_mean.bias, self.mu_bias_init)
        self.fc_log_var = nn.Linear(self.hidden_layers_sizes[-1],self.z_dim)
        nn.init.constant_(self.fc_log_var.bias, self.log_var_bias_init)

        # set up non-linearity
        if params['nonlinear_activation'] == 'relu':
            self.nonlinear_activation = nn.ReLU()
        elif params['nonlinear_activation'] == 'tanh':
            self.nonlinear_activation = nn.Tanh()
        elif params['nonlinear_activation'] == 'sigmoid':
            self.nonlinear_activation = nn.Sigmoid()
        elif params['nonlinear_activation'] == 'elu':
            self.nonlinear_activation = nn.ELU()
        elif params['nonlinear_activation'] == 'linear':
            self.nonlinear_activation = nn.Identity()
        
        if self.dropout_proba > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_proba)

    def forward(self, x):
        if self.dropout_proba > 0.0:
            x = self.dropout_layer(x)

        if self.convolve_input:
            x = x.permute(0,2,1) 
            x = self.input_convolution(x)
            x = x.view(-1,self.seq_len*self.channel_size)
        else:
            x = x.view(-1,self.seq_len*self.channel_size) 
        
        for layer_index in range(len(self.hidden_layers_sizes)):
            x = self.nonlinear_activation(self.hidden_layers[str(layer_index)](x))
            if self.dropout_proba > 0.0:
                x = self.dropout_layer(x)

        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        return z_mean, z_log_var