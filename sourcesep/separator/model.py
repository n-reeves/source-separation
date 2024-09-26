import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self,
                 io_ch: int,
                 hid_ch: int,
                 kernel: int=3,
                 dil_iter: int=0,
                 send_resid: int=True):
        """Convolutional block for TCN
        Args:
            io_ch (int): Number of input/output channels
            hid_ch (int): Number of channels passed between depth-sep conv layers in blocks
            kernel (int, optional): Kernel size for depthwise conv. Defaults to 3.
            dil_iter (int, optional): Iteration of dilation. Defaults to 0.
            send_resid (int, optional): Whether to send residual to next block. Defaults to True.
        """
        super().__init__()
        self.io_ch = io_ch
        self.hid_ch = hid_ch
        self.kernel = kernel #kernel used in depthwise seperable conv
        self.send_resid = send_resid

        #dilation is distance between elements in input used to compute one element in output
        #expoential increase in dialation allows greater temporal receptive field in less layers
        self.dilation = 2**dil_iter
        
        self.padding = self.dilation

        #activations
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

        #normalize across channel outs within conv block
        self.ln1 = nn.GroupNorm(1, self.hid_ch, eps=1e-8)
        self.ln2 = nn.GroupNorm(1, self.hid_ch, eps=1e-8)

        #first conv layer. pointwise conv
        self.block_in = nn.Conv1d(in_channels=self.io_ch  
                                 ,out_channels=self.hid_ch
                                 ,kernel_size=1)

        #depthwise conv: one filter for each input channel, out channel only connected to corresponding input channel
        self.conv_depth = nn.Conv1d(in_channels=self.hid_ch
                                    ,out_channels=self.hid_ch
                                    ,kernel_size=self.kernel
                                    ,dilation=self.dilation 
                                    ,groups=self.hid_ch #conv applies to groups of one input channel
                                    ,padding=self.padding) #casual: remove padding, only add to one side

        #output to next block
        self.block_out = nn.Conv1d(self.hid_ch, self.io_ch, 1)

        #skip connection output, same number of channels as conv block io for simplicity
        self.skip_out = nn.Conv1d(self.hid_ch, self.io_ch, 1)

    def forward(self,input):
        #residual path of block is input to next block, 
        #skip connection outs are summed and used as TCN out
        
        #pointwise conv
        out_p = self.block_in(input)
        out_p = self.prelu1(out_p)
        out_p = self.ln1(out_p)
        
        #depthwise conv
        out_p = self.conv_depth(out_p) # output = self.ln2(self.prelu2(self.dconv1d(output)[:,:,:-self.padding]))
        out_p = self.prelu2(out_p)
        out_p = self.ln2(out_p)

        #Calcuate residual to send to next block
        resid = self.block_out(out_p)
        
        skip = self.skip_out(out_p)
        
        return (resid if self.send_resid else torch.zeros_like(resid), skip)
    
    
class ConvTasNet(nn.Module):
    def __init__(self,
                 filters: int=256,
                 segment_len: float=16,  
                 stride: float=.5, 
                 tcn_io: int=128, 
                 tcn_hid: int=256,
                 conv_blocks: int=8,
                 num_tcns: int=2,
                 in_samples: int=32640, 
                 sr: int=8000 ):
        """Convolutional TasNet for source separation
        Args:
            filters (int, optional): Number of channels after encoding. Defaults to 256.
            segment_len (float, optional): Sample len of 1D conv kernel in encoder. Defaults to 16.
            stride (float, optional): Stride size in relation to L. Defaults to .5.
            tcn_io (int, optional): In/out channels in conv blocks. Defaults to 128.
            tcn_hid (int, optional): Number of channels passed between depth-sep conv layers in blocks. Defaults to 256.
            conv_blocks (int, optional): Number of blocks in TCN. Defaults to 8.
            num_tcns (int, optional): Number of TCNs. Defaults to 2.
            in_samples (int, optional): Length of each audio clip. Defaults to 32640.
            sr (int, optional): Sample rate of training data. Defaults to 8000.
        """
        super().__init__()
        self.filters = filters
        self.segment_len = segment_len
        self.stride_len = int(round(stride*self.segment_len))
        self.tcn_io = tcn_io
        self.tcn_hid = tcn_hid
        self.conv_blocks = conv_blocks
        self.num_tcns = num_tcns 
        self.in_samples=in_samples
        self.sr = sr

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
        #encoder and decoders
        self.enc = nn.Conv1d(in_channels=1, 
                         out_channels=self.filters, 
                         kernel_size=self.segment_len,  
                         stride=self.stride_len, 
                         bias=False)

        #just in case, remainder of input sample length-kernel size divided by stride
        self.dec_padding = int((self.in_samples - self.segment_len ) % self.stride_len) 
        self.dec = nn.ConvTranspose1d(in_channels=self.filters
                                      ,out_channels=1
                                      ,kernel_size=self.segment_len
                                      ,stride=self.stride_len
                                      ,padding=self.dec_padding
                                      ,bias=False)
    
        #Encoded signal to TCN input
        self.pre_tcn_norm = nn.GroupNorm(1, self.filters, eps=1e-8)
        
        self.conv_to_tcn = nn.Conv1d(in_channels=self.filters, 
                             out_channels= self.tcn_io, 
                             kernel_size=1) #used to transform enc channels to io channels for TCN blocks

        #create TCN
        self.tcns = nn.ModuleList([])
        out_resid =True
        for tcn_num in range(self.num_tcns):
            tcn = nn.ModuleList([])
            for iter in range(self.conv_blocks):
                #last block and doesn't send resid
                if iter == self.conv_blocks -1 and tcn_num == self.num_tcns -1: 
                    out_resid=False
                block = ConvBlock(io_ch=self.tcn_io
                                    ,hid_ch=self.tcn_hid
                                    ,kernel=3
                                    ,dil_iter=iter
                                    ,send_resid=out_resid)
                tcn.append(block) 
            self.tcns.append(tcn)  
        
        #Takes sum of tcn skip connection outputs and produces source masks
        self.tcn_out = nn.Sequential(nn.PReLU()
                                       ,nn.Conv1d(
                                           in_channels=self.tcn_io
                                           ,out_channels=2*self.filters #out masks for each basis signal and each speaker
                                           ,kernel_size=1
                                           ,stride=1)
                                       ,self.sig)
        
    def forward(self, x):
        #input signal dim: (Batch size, 1, clip_length)
        #output dim: (batch,speakers,1,clip_length)
        batch_size = x.shape[0]
        self.in_samples = x.shape[2]
        #produce mixture weight of encoded signals. Out dim: (batch, N, L) (L is the number of encoded time segments for the clip)
        #store for later
        x_enc = self.relu(self.enc(x))

        time_segments = x_enc.shape[2]

        #norm
        x_tcn = self.pre_tcn_norm(x_enc)
        
        x_tcn = self.conv_to_tcn(x_tcn)

        #init array to add skip connections and apply TCN to produce masks
        skip_sum = torch.zeros((batch_size, self.tcn_io, time_segments), device=x_tcn.device)
        for tcn in self.tcns:
            for block in tcn:
                (res_out,skip_out) = block(x_tcn)
                x_tcn = x_tcn + res_out
                skip_sum = skip_sum + skip_out

        #Basis signal masks. dim: (batch, sources, N, L)
        masks = self.tcn_out(skip_sum).reshape((batch_size, 2, self.filters, -1))

        #add speaker axes to encoded signal and multiply by speaker masks
        x_enc = x_enc.unsqueeze(1).repeat(1,2,1,1)
        x_enc = x_enc * masks

        #conv transpose expects (batch,Channels,dims)
        #treat each source as individual batch and then apply conv (batch*speakers,N,L)
        x_enc = x_enc.reshape((batch_size*2,self.filters,-1))

        #decode to (batch*speakers,1,clip_length) and reshape to (batch,speakers,1,clip_lengt)
        x_dec = self.dec(x_enc)
        x_dec = x_dec.reshape(batch_size,2,1,-1)
        return x_dec

