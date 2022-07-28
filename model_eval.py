import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)
            
class Non_Local_Block(torch.nn.Module):
    def __init__(self, embed_dim,mid_dim,dropout_ratio):
        super().__init__()
        embed_dim = 2048
        mid_dim = 256
        self.Theta = nn.Sequential(
            nn.Conv1d(embed_dim,mid_dim, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.Phi =   nn.Sequential(
            nn.Conv1d(embed_dim,mid_dim, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.Gamma = nn.Sequential(
            nn.Conv1d(embed_dim,mid_dim, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.Conv_out = nn.Sequential(
            nn.Conv1d(mid_dim,embed_dim, 1, padding=0),nn.LeakyReLU(0.2))#,nn.Dropout(dropout_ratio))
    def forward(self,x):
        theta = self.Theta(x) #B,C//4,T
        phi = self.Phi(x)
        gamma = self.Gamma(x)
        phi = phi.permute(0,2,1) #B,T,C//4
        gamma = gamma.permute(0,2,1) #B,T,C//4
        middle1 = torch.matmul(phi,theta) #B,T,T
        middle1 /= 32
        middle1 = middle1.softmax(dim=-1)
        y = torch.matmul(middle1,gamma)#B,T,C//4
        y = y.permute(0,2,1)#B,C//4,T
        output = x + self.Conv_out(y)
        return output



class Base0(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim = 1024
        n_feature = 2048
        dropout_ratio=args['opt'].dropout_ratio
        
        self.att_r = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5),
                                      nn.Conv1d(512, 512, 3, padding=1),nn.LeakyReLU(0.2), 
                                      nn.Conv1d(512, 1, 1),nn.Dropout(0.5),nn.Sigmoid())
        self.att_f = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5),
                                      nn.Conv1d(512, 512, 3, padding=1),nn.LeakyReLU(0.2), 
                                      nn.Conv1d(512, 1, 1),nn.Dropout(0.5),nn.Sigmoid())
       
        self.fusion1 = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio)
            )
        self.fusion3 = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio)
            )
        self.classifier = nn.Conv1d(n_feature, n_class+1, 1)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        x = inputs.transpose(-1, -2) #X B,D,T
        x_r = inputs[:,:,:1024].transpose(-1,-2)
        x_f = inputs[:,:,1024:].transpose(-1,-2)
        
        a_r = self.att_r(x_r)
        a_f = self.att_f(x_f)
        att = (a_r + a_f)/2

        A_trans = att.clone().detach()
        fusion_feat1 = self.fusion1(x)
        fusion_feat =fusion_feat1
        fusion_feat_o = self.fusion3(fusion_feat)
        x_cls = self.classifier(fusion_feat_o) 

        return {
            'feat':fusion_feat.transpose(-1, -2), 
            'cas':x_cls.transpose(-1, -2), 
            'atn':att.transpose(-1, -2),
            'atn_rgb':a_r.transpose(-1, -2),
            'atn_flow':a_f.transpose(-1, -2),
            'A_trans':A_trans
            }



class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
                                       nn.Conv1d(1024, 512, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),nn.LeakyReLU(0.2), 
                                       nn.Conv1d(512, 1, 1),nn.Dropout(0.5),
                                       nn.Sigmoid())
    def forward(self,feat):
        atn = self.attention(feat)
        return atn

class BiSCC(torch.nn.Module):
    def __init__(self, num_features, num_classes,**args):
        super().__init__()
        embed_dim=2048
        dropout_ratio=args['opt'].dropout_ratio
        self.atn_r = Attention()
        self.atn_f = Attention()
        self.emb1 = nn.Sequential(nn.Conv1d( num_features, num_features, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.emb2  = Non_Local_Block( num_features,256,dropout_ratio)
        self.classifier = nn.Sequential(
            nn.Conv1d( num_features* 2  , num_features, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.7), 
            nn.Conv1d( num_features, num_classes+1, 1)
            )
        self.apply(weights_init)
        
    def forward(self, inputs, is_training=True, **args):
        feat = inputs.permute(0,2,1)
        b,c,n=feat.size()
        atn_rgb = self.atn_r(feat[:,:1024,:])
        atn_flow = self.atn_f(feat[:,1024:,:])
        atn = (atn_rgb+atn_flow)/2
        feat1 = self.emb1(feat)
        feat2 = self.emb2(feat)
        final_feat =  torch.cat((feat1,feat2),1)
        x_cls = self.classifier(final_feat) 
        
        return {
            'feat':final_feat.permute(0,2,1), 
            'cas':x_cls.permute(0,2,1), 
            'atn':atn.permute(0,2,1),
            ' atn_rgb': atn_rgb.permute(0,2,1),
            ' atn_flow': atn_flow.permute(0,2,1),
            }
