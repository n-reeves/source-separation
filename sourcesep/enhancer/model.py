import torch
from torch import nn

class ComplexBatchNorm(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.ch = ch #number of channels to normalize
        
        #scaling parameters
        self.gamma_r = torch.nn.Parameter(torch.full((ch,1),1/torch.sqrt(torch.tensor(2)))) #one weight per element in channel real/imag cov matrix
        self.gamma_i = torch.nn.Parameter(torch.full((ch,1),1/torch.sqrt(torch.tensor(2)))) #one weight per element in channel real/imag cov matrix
        self.gamma_ir = torch.nn.Parameter(torch.zeros((ch,1))) #one weight per element in channel real/imag cov matrix

        #shift parameters
        self.beta_r = torch.nn.Parameter(torch.zeros((ch,1)))
        self.beta_i = torch.nn.Parameter(torch.zeros((ch,1)))
        
    def forward(self, x):
        #in dim (B,ch, H, W,2)
        #out dim (B,ch, H, W,2)
        b = x.shape[0]
        h = x.shape[2]
        w = x.shape[3]
        
        #calc means over each real/im part indp for each feature and center
        x = x.permute(1,4,0,2,3) #(ch,2,B H,W)
        x_mu = torch.mean(x, dim=(2,3,4),keepdim=True)#(ch,2,1 1,1), mean for each real and each im part of each ch
        x_cent = x - x_mu

        
        #calc per channel 2x2 real/im cov matrix
        #for each channel 2,elements in rep * transpose to get 2,2 matrix of dot products permutations between 
        #real and complex values for each channel. divide the 2x2 mats by the total num of elements in each channel to get Cov Mat
        x_cent = x_cent.reshape(self.ch, 2, -1) #(ch,2,B*H*W)
        x_cent_t = x_cent.permute(0,2,1)
        n = x_cent.shape[2]

        x_cov = torch.bmm(x_cent, x_cent_t)/n
        
        #find inverse squre root of x_cov
        var_r = x_cov[:,0,0] #all matrices have dim :(ch,) 
        var_i = x_cov[:,1,1]
        cov_ir = x_cov[:,0,1]

        #taken directly from https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py
        #Used to produce square root of inverse of covariance matrix
        tau = var_r + var_i

        #determinant of the 2x2 mat, used for numeric stability over torch.det
        #cov matrix shoudl always have det >= 0 because it is symmetric positive def
        #but rounding sometimes produces negative values very close to zero
        det = var_r * var_i - cov_ir**2 + 1e-8
        
        det_sqrt = torch.sqrt(det) #determinant of square root matrix
        t = torch.sqrt(tau + 2 * det_sqrt) 

        inverse_st = 1.0 / (det_sqrt * t + 1e-8)
        ism_r = (var_i + det_sqrt) * inverse_st
        ism_i = (var_r + det_sqrt) * inverse_st
        ism_ir = -cov_ir * inverse_st

        #apply inverted cov mat to centered data using the following formula to produce normed real and imaginary values
        ism_r = ism_r.view((-1,1)) * self.gamma_r 
        ism_i = ism_i.view((-1,1)) * self.gamma_i
        ism_ir = ism_ir.view((-1,1)) *  self.gamma_ir
        
        x_out = torch.zeros_like(x_cent) #(ch,2,B*H*W)
        x_out[:,0,:] = ism_r * x_cent[:,0,:] + ism_ir*x_cent[:,1,:] + self.beta_r
        x_out[:,1,:] = ism_ir * x_cent[:,0,:] + ism_i*x_cent[:,1,:] + self.beta_i

        x_out = x_out.reshape(self.ch,2,b,h,w).permute(2,0,3,4,1) #(ch,2,B*H*W) -> (B,ch, H, W,2)
        return x_out
    

class ComplexConv2D(nn.Module):
    def __init__(self, in_ch: int, 
                 out_ch: int, 
                 kernal: torch.tensor, 
                 stride:torch.tensor, 
                 pad:torch.tensor):
        """2D convolution for complex inputs

        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            kernal (torch.tensor): size of convolutional kernel
            stride (torch.tensor): stride of convolution
            pad (torch.tensor): padding of convolution
        """
        super().__init__()
        #Convolutions for real and imaginary parts of complex data
        self.real_c = torch.nn.Conv2d(in_channels=in_ch
                                      ,out_channels=out_ch
                                      ,kernel_size=kernal
                                      ,stride=stride
                                      ,padding=pad)
        self.im_c = torch.nn.Conv2d(in_channels=in_ch
                                      ,out_channels=out_ch
                                      ,kernel_size=kernal
                                      ,stride=stride
                                      ,padding=pad)

    def forward(self,x):
        #in dim (B,in_ch, H, W,2)
        #out dim (B,out_Ch, Hout, Wout,2)
        real_pt = x[:,:,:,:,0]
        im_pt = x[:,:,:,:,1]

        #complex conv on h = x +yi with complex filter W = A + Bi
        #Wh = (A*x - B*y) + i(B*x + A*y)
        #x_re is the set of real valued channels
        x_re = self.real_c(real_pt) - self.im_c(im_pt) #A*x-B*y
        x_im = self.real_c(im_pt) + self.im_c(real_pt) #i(B*x + A*y)

        x = torch.stack((x_re, x_im), dim=-1)
    
        return x


class ComplexConvBlock(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 kernal: torch.tensor=torch.tensor([5,5]), 
                 stride: torch.tensor=torch.tensor([2,2]), 
                 pad: torch.tensor=torch.tensor([2,2]), 
                 resid: bool=False, 
                 first_block:bool=False):
        """
        Complex convolutional block
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            kernal (torch.tensor, optional): size of convolutional kernel. Defaults to torch.tensor([5,5]).
            stride (torch.tensor, optional): stride of convolution. Defaults to torch.tensor([2,2]).
            pad (torch.tensor, optional): padding of convolution. Defaults to torch.tensor([2,2]).
            resid (bool, optional): whether to learn residual. Defaults to False.
            first_block (bool, optional): whether block is first in network. Defaults
        """
        super().__init__()
        #block is made up of complex conv, complex batch norm, and leaky relu
        self.conv = ComplexConv2D(in_ch, out_ch, kernal, stride, pad)
        self.complex_bn = ComplexBatchNorm(ch=out_ch)
        self.act = nn.LeakyReLU()
        self.first_block = first_block
        self.resid = resid
    
    def forward(self,x):
        #in dim (B,in_ch, H, W,2)
        #out dim (B,out_Ch, Hout, Wout,2)

        #first block only convolves
        if self.first_block:
            x = self.conv(x)

        else:
            #apply relu as usual as CReLU described in Trabelsi, 2018 
            #for a complex num c = x +iy is ReLU(x) +iReLU(y)
            x_in = self.act(x)
    
            x = self.conv(x_in)

            #conv blocks in skip connection learn residual
            if self.resid:
                x = x + x_in
           
            x = self.complex_bn(x)
        return x


class SkipLayers(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 chan: int, 
                 kernal: torch.tensor=torch.tensor([3,3]), 
                 stride: torch.tensor=torch.tensor([1,1]),
                 pad: torch.tensor=torch.tensor([1,1])):
        super().__init__()

        #create skip connection blocks
        blocks = []
        for layer in range(num_layers):
            #network params produce io dimensions equal to input dimensions
            block = ComplexConvBlock(chan, chan, kernal=kernal, stride=stride, pad=pad, resid=True)
            blocks.append(block)

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        #in dim: (B,chan, H, W,2)
        #out dim: (B,chan, H, W,2)
        x = self.network(x)
        return x


class ComplexConvTrans2D(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 kernal: torch.tensor, 
                 stride: torch.tensor, 
                 out_pad: torch.tensor):
        super().__init__()
        #similar logic extends to complex transposition, two blocks that transpose real the real and imaginary parts
        self.real_ct = torch.nn.ConvTranspose2d(in_channels=in_ch
                                                ,out_channels=out_ch
                                                ,kernel_size=kernal
                                                ,stride=stride
                                                ,output_padding=out_pad)
        
        self.im_ct = torch.nn.ConvTranspose2d(in_channels=in_ch
                                                ,out_channels=out_ch
                                                ,kernel_size=kernal
                                                ,stride=stride
                                                ,output_padding=out_pad)
    
    def forward(self,x):
        #in dim (B,in_ch, H, W,2)
        #out dim (B,out_Ch, Hout, Wout,2)
        real_pt = x[:,:,:,:,0]
        im_pt = x[:,:,:,:,1]

        #complex transpose: project back to h = x +yi from Wh = Ah +iBh = (A*x - B*y) + i(B*x + A*y)
        #conv takes (x,y) -> (u,v)= (A*x - B*y, B*x + A*y)
        #x is convolved by A to real part u and by B to imag part v 
        #y is concolved by -B to real part u and by A to imag part v
        x_re = self.real_ct(real_pt) + self.im_ct(im_pt) # ATu + BTv 
        x_im = self.real_ct(im_pt) - self.im_ct(real_pt) # ATv - BTu

        x = torch.stack((x_re, x_im), dim=-1)
        return x


class ComplexConvTransBlock(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 kernal: torch.tensor, 
                 stride: torch.tensor, 
                 out_pad: torch.tensor, 
                 last_block: bool=False):
        super().__init__()

        self.conv_t = ComplexConvTrans2D(in_ch, out_ch, kernal, stride, out_pad)
        self.complex_bn = ComplexBatchNorm(ch=out_ch)
        self.act = nn.LeakyReLU()
        self.last_block = last_block

    def forward(self,x):
        #in dim (B,in_ch, H, W,2)
        #out dim (B,out_Ch, Hout, Wout,2)
        
        x = self.act(x)

        x = self.conv_t(x)

        #last decoder block is not normalized
        if not self.last_block:
            x = self.complex_bn(x)
        
        return x
    

class ComplexSkipConvNet(nn.Module):
    def __init__(self, 
                 net_params: dict, 
                 alt_mask: bool=False):
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.skip_cons = nn.ModuleList([])
        
        self.tr_pad = [] #contains padding for transpose layers
        self.tanh = nn.Tanh()
        self.alt_mask=alt_mask

        #build U-net
        inp_chan = 1
        #build encoder and skip networks
        for i,param_dict in enumerate(net_params['enc']):
            if i == 0:
                first_block = True 
            else:
                first_block = False 
            
            conv_block = ComplexConvBlock(in_ch=inp_chan
                                            ,out_ch=param_dict['out_ch']
                                            ,kernal=param_dict['k']
                                            ,stride=param_dict['s']
                                            ,pad=param_dict['p']
                                            ,first_block=first_block)
            self.encoder.append(conv_block)

            #skip network
            skip_con = SkipLayers(param_dict['sk_bl']
                                  ,param_dict['out_ch']
                                  ,kernal=net_params['sk']['k']
                                  ,stride=net_params['sk']['s']
                                  ,pad=net_params['sk']['p'] )
            self.skip_cons.append(skip_con)
            
            inp_chan = param_dict['out_ch'] #set input channel for next encoder block
            
        #build decoder
        for i, param_dict in enumerate(net_params['dec']):
            if i == len(param_dict) -1:
                last_block = True 
            else:
                last_block = False 
                        
            convt_block = ComplexConvTransBlock(in_ch=inp_chan
                                              ,out_ch=param_dict['out_ch']
                                              ,kernal=param_dict['k']
                                              ,stride=param_dict['s']
                                              ,out_pad=param_dict['op']
                                              ,last_block=last_block)
            
            self.decoder.append(convt_block)
            inp_chan = param_dict['out_ch']*2 #U net decoder layers take in skip connections as additional channels
            
        
    def forward(self,x):
        #in dim: (B,1,F,T,2)
        #out dim (B,1,F,T)
        inp = x
        skip_outs = []
        #encode inputs
        for i, enc_block in enumerate(self.encoder):
            x = enc_block(x)
            
            #store intermediate outputs from each encoder block before the last
            if i < len(self.encoder) -1:
                skip_net = self.skip_cons[i] 
                skip_out = skip_net(x)
                
                skip_outs.append(skip_out) #store the intermediate outputs from all but the last layer

        #decode the encodings and skip outs
        for i, dec_block in enumerate(self.decoder):
            if i > 0: #first decoder block only recieves output from last layer
                skip_out = skip_outs[-i] #
                x = torch.cat((x, skip_out), dim=1)
            x = dec_block(x)

        #masking method provided by choi, not evaluated in project
        if self.alt_mask:
            #masking method from choi, 2019
            #produce bounded magnitude and phase masks
            x = torch.view_as_complex(x)
            x_mag = torch.abs(x)
            mask_mag = self.tanh(x_mag)
            mask_phase = x/x_mag
    
            mask = mask_mag*mask_phase
            
            out = mask*x
        else:
            #Ephrat et al 2018 bounded sigmoid (using tanh instead)
            #activation function applied to real and imaginary coefficients masks
            #simple, but masks can produce complex numbers that have magnitude > 1
            x = self.tanh(x)
            out = inp*x
            out = torch.view_as_complex(out)
        return out
    
    

    