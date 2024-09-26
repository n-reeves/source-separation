import torch
from torch import nn

from sourcesep.enhancer.model import ComplexSkipConvNet
from sourcesep.separator.model import ConvTasNet

class SourceSeparator(nn.Module):
    def __init__(self,
                 #U-Net Parameters
                 unet_params: dict,
                 #TasNet Parameters
                 filters: int=256,
                 segment_len: float=16,
                 stride: float =.5, 
                 tcn_io: int = 128, 
                 tcn_hid: int = 256,
                 conv_blocks: int=8, 
                 num_tcns: int=2,
                 in_samples: int=32640,
                 #General parameters
                 sep_net: bool=True,
                 sr: int=8000,
                 hop_len_s: float= .016,
                 win_s: float= .064):
        """Joint model for source separation

        Args:
            unet_params (dict): Dictionary of parameters for U-Net
            filters (int, optional): Number of channels after encoding. Defaults to 256.
            segment_len (float, optional): Sample len of 1D conv kernel in encoder. Defaults to 16.
            stride (float, optional): Stride size in relation to L. Defaults to .5.
            tcn_io (int, optional): In/out channels in conv blocks. Defaults to 128.
            tcn_hid (int, optional): Number of channels passed between depth-sep conv layers in blocks. Defaults to 256.
            conv_blocks (int, optional): Number of blocks in TCN. Defaults to 8.
            num_tcns (int, optional): Number of TCNs. Defaults to 2.
            in_samples (int, optional): Length of each audio clip. Defaults to 32640.
            sep_net (bool, optional): Whether to use separator. Defaults to True.
            sr (int, optional): Sample rate of training data. Defaults to 8000.
            hop_len_s (float, optional): Hop length in seconds. Defaults to .016.
            win_s (float, optional): Window size in seconds. Defaults to .064.
        """
        super().__init__()
        self.sep_net = sep_net
        self.sr = sr
        self.hop_len_s = hop_len_s
        self.win_s = win_s
        self.hop_len_sam = int(round(self.hop_len_s*self.sr))
        self.win_len_sam = int(round(self.win_s*self.sr))
        self.in_samples = in_samples
        
        #register widnow as buffer so it is moved to device
        window_f = torch.hann_window(self.win_len_sam,)
        self.register_buffer('window_f', window_f)
        
        self.u_net = ComplexSkipConvNet(unet_params)
        
        self.tas_net = ConvTasNet(filters=filters 
                                 ,segment_len=segment_len  
                                 ,stride=stride
                                 ,tcn_io=tcn_io 
                                 ,tcn_hid=tcn_hid
                                 ,conv_blocks=conv_blocks
                                 ,num_tcns=num_tcns
                                 ,in_samples=in_samples 
                                 ,sr=sr)

    def toggle_sep(self):
        self.sep_net =  not self.sep_net
        print('Separator active: {}'.format(self.sep_net))

    def batch_istft(self, x):
        #defined inside of object as istft needs to be stored on same device
        
        #input: complex tensor with dim (B, 1, f_bins, h_bins)
        #out: real valued tensor with dim (B, 1, utter samples)
        x = x.squeeze(1)
        wavs = torch.istft(x
                          ,n_fft=self.win_len_sam
                          ,window=self.window_f
                          ,hop_length=self.hop_len_sam)
        return wavs.unsqueeze(1)
    
    def forward(self, x):
        #input dim (B,1,F,T,2)
        out_dict = {}
        out_m1 = self.u_net(x) #returns (B,1,F,T,2)

        out_m1_clean = self.batch_istft(out_m1) #returns (B,1,samples)
        
        out_dict['m1_out'] = out_m1_clean

        if self.sep_net:
            out_m2 = self.tas_net(out_m1_clean) #returns (B,2,1,samples)
            out_dict['m2_out'] = out_m2

        return out_dict