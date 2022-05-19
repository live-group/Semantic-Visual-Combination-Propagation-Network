import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import BatchNorm1d


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight,gain=0.5)
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class encoder_template(nn.Module):    #编码器

    def __init__(self,input_dim,latent_size,hidden_size_rule, hidden_size_rule_de, output_dim):
        super(encoder_template,self).__init__()

        #2048, 256,  1024,  256, 102
        self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]  #
        self.feature_encoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),
                                              nn.Linear(self.layer_sizes[1],latent_size),
                                             )  #2048, 1024, 256
        self.apply(weights_init)
        #self.to(device)
        self.decoder = decoder_template(latent_size, output_dim, hidden_size_rule_de)


    def forward(self,x):
        h = self.feature_encoder(x)
        h= self.decoder(h)
        return h


class decoder_template(nn.Module):  #解码器

    def __init__(self,input_dim,output_dim,hidden_size_rule):  #256,  256, 102
        super(decoder_template,self).__init__()
        self.layer_sizes = [input_dim, hidden_size_rule[0] , output_dim]
        #self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),nn.Linear(self.layer_sizes[1],output_dim))
        self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),
                                             nn.Linear(self.layer_sizes[1],output_dim),
                                              )
        self.apply(weights_init)
        #self.to(device)
    def forward(self,x):
        return self.feature_decoder(x)      
        
  
'''



class encoder_template(nn.Module):    #编码器

    def __init__(self,input_dim,latent_size,hidden_size_rule):
        super(encoder_template,self).__init__()

        if len(hidden_size_rule)==2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule)==3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1] , latent_size]
        modules = []
        for i in range(len(self.layer_sizes)-2):  #1或者2
            modules.append(nn.Linear(self.layer_sizes[i],self.layer_sizes[i+1]))
            modules.append(nn.ReLU())
            #modules.append(nn.BatchNorm1d(self.layer_sizes[i+1]))
            #modules.append(nn.LeakyReLU(0.2, True))
        self.feature_encoder = nn.Sequential(*modules)
        #self.lrelu = nn.LeakyReLU(0.2, True)
        self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self.apply(weights_init)
        #self.to(device)


    def forward(self,x):
        h = self.feature_encoder(x)
        mu =  self._mu(h)
        logvar = self._logvar(h)
        return mu, logvar
        

class decoder_template(nn.Module):  #解码器

    def __init__(self,input_dim,output_dim,hidden_size_rule):
        super(decoder_template,self).__init__()
        self.layer_sizes = [input_dim, hidden_size_rule[-1] , output_dim]
        #self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),nn.Linear(self.layer_sizes[1],output_dim))
        self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),nn.Linear(self.layer_sizes[1],output_dim))
        self.apply(weights_init)
        #self.to(device)
    def forward(self,x):
        return self.feature_decoder(x)
        
class Decoder(nn.Module):
    def __init__(self, in_features, out_features, p=0.5):
        super(Decoder, self).__init__()

        self.fc_1 = nn.Linear(in_features=in_features, out_features=out_features)
        #self.bn_1 = BatchNorm1d(out_features)
        self.bn_1 = nn.BatchNorm1d(num_features=out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.drop_1 = nn.Dropout(p=p)

        self.fc_2 = nn.Linear(in_features=out_features, out_features=out_features)
        # self.bn_2 = nn.BatchNorm1d(num_features=in_features//4)
        # self.drop_2 = nn.Dropout(p=p)

        # self.fc_last = nn.Linear(in_features=in_features//4, out_features=out_features)

    def forward(self, feature):

        fc_1_out = self.fc_1(feature)
        # fc_1_out = self.drop_1(fc_1_out)
        #fc_1_out = self.bn_1(fc_1_out)
        fc_1_out = F.relu(fc_1_out)

        fc_2_out = self.fc_2(fc_1_out)
        # fc_2_out = self.drop_2(fc_2_out)
        # fc_2_out = self.bn_2(fc_2_out)
        # fc_2_out = F.relu(fc_2_out)

        # fc_last_out = self.fc_last(fc_2_out)

        return fc_2_out





#CVAE_decoder(s_features=300, c_features=2048, h_features=256,out_features=300, drop_out=self.drop_out)  
     
class CVAE_decoder(nn.Module):
    def __init__(self, s_features, c_features, h_features, out_features, drop_out=0.5):
        super(CVAE_decoder, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(in_features=s_features, out_features=s_features),
            nn.ReLU())

        self._enc_mean = nn.Linear(in_features=s_features + c_features, out_features=h_features)
        self._enc_log_sigma = nn.Linear(in_features=s_features + c_features, out_features=h_features)

        self.decoder = Decoder(in_features=h_features + c_features, out_features=out_features, p=drop_out)

    def _sample_latent(self, h_enc):

        mean = self._enc_mean(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)

        sigma = torch.exp(0.5 * log_sigma)
        std_z = torch.randn(sigma.shape).float().cuda()

        return mean + sigma * std_z, mean, sigma

    def forward(self, x, semantic):
        semantic = self.map(semantic)
        z, z_mean, z_sigma = self._sample_latent(torch.cat((semantic, x), dim=1))
        z_c = torch.cat((z, x), dim=1)
        output= self.decoder(z_c)

        return output, z_mean, z_sigma



class Latent_Loss(nn.Module):
    def __init__(self):
        super(Latent_Loss, self).__init__()

    def forward(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.sum(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
'''