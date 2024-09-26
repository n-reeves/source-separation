import torch
from torch.utils.data import Dataset
import torch_audiomentations as ta

from sourcesep.utils import batch_crop, batch_stft

import os
import numpy as np

class SourceSepDS(Dataset):
    def __init__(self,
                 audio_dir: str, 
                 apply_ir: bool=True, 
                 ir_determ: bool=False,
                 ir_p: float=1,
                 sr: int=8000, 
                 hop_len_s: float=.016,
                 win_s: float=.064,
                 inp_secs: float=4.08,
                 segment_len: float=16.0,
                 sep_net: bool=True,
                 rand_crop: bool=True):
        """Dataset class for source separation task

        Args:
            audio_dir (str): path to directory containing audio files
            apply_ir (bool, optional): apply impulse responses to audio files. Defaults to True.
            ir_determ (bool, optional): apply impulse responses in order. Defaults to False.
            ir_p (float, optional): probability of applying ir. Defaults to 1.
            sr (int, optional): sample rate of audio. Defaults to 8000.
            hop_len_s (float, optional): hop length of stft in seconds. Defaults to .016.
            win_s (float, optional): window length of stft in seconds. Defaults to .064.
            inp_secs (float, optional): length of input segments. Defaults to 4.08.
            segment_len (float, optional): length of 1d convolutions in separation network encoder. Defaults to 16.0.
            sep_net (bool, optional): True if dataset is to be used for training the enhancer and source separator, false if just enhancer. Defaults to True.
            rand_crop (bool, optional): specify whether or not to randomize start index of crop. Defaults to True.
        """
        
        self.sr=sr
        self.inp_secs = inp_secs #length of input segments
        self.sep_net = sep_net #True if dataset is to be used for the de-reverb and source separator, false if just de reverb
        self.rand_crop = rand_crop
        
        #parameters for STFT used in denoise/dereverb network input
        self.hop_len_s = hop_len_s
        self.win_len_s = win_s

        #length of 1d convolutions in separation network encoder
        self.sep_encode_len = segment_len 

        #calculate the sample length of each input segment(utterance)
        #get number of samples in each segment and multiply by rounded number of segements in each clip
        clip_samples = self.sr * self.inp_secs
        num_sep_segments = int(np.ceil(clip_samples/self.sep_encode_len))
        self.utterance_samples = num_sep_segments*self.sep_encode_len
        
        #set audio directory paths
        self.audio_dir = audio_dir
        self.mix_noise_dir = os.path.join(self.audio_dir ,'mix_noise')
        self.mix_clean_dir = os.path.join(self.audio_dir ,'mix_clean')
        self.s1_dir = os.path.join(self.audio_dir ,'s1')
        self.s2_dir = os.path.join(self.audio_dir ,'s2')

        #list of directories each batch loads audio from
        if self.sep_net:
            self.dir_load = [self.mix_noise_dir, self.mix_clean_dir, self.s1_dir, self.s2_dir]
        else: #speed up training by not loading sources when only training denoiser
            self.dir_load = [self.mix_noise_dir, self.mix_clean_dir]

        #load filenames
        mix_noise_files = os.listdir(self.mix_noise_dir)
        mix_clean_files = set(os.listdir(self.mix_clean_dir))
        s1_files = set(os.listdir(self.s1_dir))
        s2_files = set(os.listdir(self.s2_dir))
        
        #Remove non-wav files from noisy mixture dir, only include files that have all variants present
        ds_files = []
        clean_mix_noise = [x for x in mix_noise_files if x[-4:] =='.wav']
        for filename in clean_mix_noise:
            if filename in mix_clean_files and filename in s1_files and filename in s2_files:
                ds_files.append(filename)
            
        self.files = np.array(ds_files, dtype=str)

        #impulse response transformation
        self.ir_dir = os.path.join(self.audio_dir ,'ir')
        self.apply_ir = apply_ir #boolean indicating whether or not to apply impulse responses
        self.ir_transform = ta.ApplyImpulseResponse(ir_paths = self.ir_dir
                                                    ,p = ir_p
                                                    ,compensate_for_propagation_delay=True
                                                    ,sample_rate=8000)

        #used in deterministic application of ir
        #iterates through your ir files applies them in order repeatedly
        #used for testing with dataloader shuffle=False and rand_crop=False
        self.ir_determ = ir_determ
        self.ir_counter = 0
        self.ir_filenames = [x for x in os.listdir(self.ir_dir) if x[-4:] =='.wav']
        self.ir_len = len(self.ir_filenames)
        
        
    def __len__(self):
        return self.files.shape[0]
    
    def __getitem__(self, idx):
        #if batch is retrieved using tensor
        if torch.is_tensor(idx):
            idx = idx.numpy()

        filenames = self.files[idx]
        #if idx is a single value, turn filenames into a 1d array for compatibility with batch_crop
        if isinstance(filenames, str):
            filenames = np.array([filenames])
        wav_dict = batch_crop(filenames
                              ,self.dir_load
                              ,sr=self.sr
                              ,crop_len=self.utterance_samples
                              ,rand_crop=self.rand_crop)

        #create output dictionary
        out = {}
        
        #clean input and output signals: dim (1, clip_samples) (batch dim is created when emitting eachind record by dataloader)
        signal_mix_noise = torch.cat(wav_dict[self.mix_noise_dir],dim=0)
        signal_mix_clean = torch.cat(wav_dict[self.mix_clean_dir],dim=0)

        signal_mix_input = signal_mix_noise
        #reverberated input signals
        if self.apply_ir:
            if self.ir_determ:
                #iterate through ir files in order. if at the end of the list, reset counter
                #create ir_transform object for that points to specific wav
                ir_filename = self.ir_filenames[self.ir_counter]
                ir_dir = os.path.join(self.ir_dir , ir_filename)
                ir_transform = ta.ApplyImpulseResponse(ir_paths = ir_dir
                                                    ,p = 1
                                                    ,compensate_for_propagation_delay=True
                                                    ,sample_rate=8000)
                
                signal_mix_input = ir_transform(signal_mix_input.unsqueeze(0)).squeeze(0)

                self.ir_counter += 1
                if self.ir_counter == self.ir_len:
                    self.ir_counter = 0
                
            else:
                #randomly apply ir in ir directory
                signal_mix_input = self.ir_transform(signal_mix_input.unsqueeze(0)).squeeze(0)

        #stft of input signals
        #batch stft expects (b,1,utterance_samples)
        stft_mix_input = batch_stft(signal_mix_input.unsqueeze(1)
                                    ,sr=self.sr
                                    ,hop_len_s=self.hop_len_s
                                    ,win_s=self.win_len_s).squeeze(0)
        
        out['signal_mix_input'] = signal_mix_input #speaker mixes that are inputs to model 1: dim (B,1,utter_samples)
        out['signal_mix_noise'] = signal_mix_noise #non-reverberated speaker mixtures: dim (B,1,utter_samples)
        out['signal_mix_clean'] = signal_mix_clean #clean speaker mixture : (B,1,utter_samples)
        out['stft_mix_input'] = stft_mix_input #stft of input speaker mixture returrns (B,1,F,T,2)

        if self.sep_net:
            s1 = torch.cat(wav_dict[self.s1_dir],dim=0).unsqueeze(1) #during emit: (1,1 utter samples), in output dict: (B,1,utter_samples)
            s2 = torch.cat(wav_dict[self.s2_dir],dim=0).unsqueeze(1)
            sources = torch.cat((s1,s2), dim=0)
            out['sources'] = sources #returns (B,2,1,utter_samples)
            
        return out