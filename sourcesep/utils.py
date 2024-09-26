import torch
import torchaudio
import torchaudio.functional as taF

import random
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

def load_wav(filepath: str,
             sr: int = 8000):
    """
    Load a wav file and resample it to the desired sampling rate

    Args:
        filepath (str): path to the wav file
        sr (int, optional): sample rate . Defaults to 8000.

    Returns:
        torch tensor: torch tensor of the audio file
    """
    audio, nat_sr = torchaudio.load(filepath) #dim: ch x samples 
    if sr is not None and sr != nat_sr :
        audio = taF.resample(audio, nat_sr, sr)
    return audio


def write_wav(filepath: str, 
              audio: torch.tensor, 
              sr: int):
    """save a torch tensor as a wav file

    Args:
        filepath (str): path to save the wav file
        audio (torch.tensor): torch tensor of the audio file
        sr (int): sample rate of the torch tensor
    """
    torchaudio.save(filepath, audio, sr)

def batch_crop(filenames: np.ndarray, 
               dirs: list, 
               sr: int=8000, 
               crop_len: int=32000,
               rand_crop: bool=True):
    """
    Crop audio files to a fixed length
    
    Args:
        filenames (np.ndarray): numpy array of filenames
        dirs (list): list of directories the audio files should be retrieved from
        sr (int, optional): sample rate of audio. Defaults to 8000.
        crop_len (int, optional): length of the cropped audio segments in samples. Defaults to 32000.
        rand_crop (bool, optional): specify whether or not to randomize start index of crop. Defaults to True.

    Returns:
        dict: contains keys as path to audio files, values as array of (1,crop_len) torch tensors
    """
    
    #initialize output dictionary
    cropped_files = {}
    for dir in dirs: 
        cropped_files[dir] = []

    #iterate through each file and each directory using the same indices for each audio clip
    for filename in filenames:
        ind_set = False #mark if random crop indices have been set yet
        for dir in dirs:
            path = os.path.join(dir,filename)
            wav = load_wav(path,sr)
            wav_len = wav.shape[1]

            #if audio file is shorter than crop size, add zero padding
            if wav_len < crop_len:
                dif = crop_len - wav_len
                pad = torch.zeros((1,dif))
                wav = torch.cat((wav,pad),dim=1)
                wav_len = crop_len

            if not ind_set:
                if rand_crop:
                    start_ind = random.randint(0, wav_len-crop_len)
                else:
                    start_ind = 0
                
                end_ind = start_ind + crop_len
                ind_set=True

            wav_crop = wav[:,start_ind:end_ind]
            cropped_files[dir].append(wav_crop)
            
    return cropped_files


def batch_stft(data: torch.tensor,
               sr: int=8000, 
               hop_len_s: float=.016, 
               win_s: float=.064):
    """Compute the short time fourier transform of a batch of audio files

    Args:
        data (torch.tensor): input audio data. dim: (B, 1, segment length)
        sr (int, optional): sample rate of the audio files . Defaults to 8000.
        hop_len_s (float, optional): hop length in seconds. Defaults to .016.
        win_s (float, optional): window length in seconds of STFT. Defaults to .064.

    Returns:
        torch.tenor: complex STFT coefficients of the input audio data. dim: (B, 1, f_bins, h_bins, 2)
    """
    hl_sam = int(round(hop_len_s*sr))
    wl_sam = int(round(win_s*sr))

    data = data.squeeze(1)
    
    #return_complex provides extra dim that contains real vals corresponding to real and im parts
    stft_coefs = torch.stft(data
                          ,n_fft=wl_sam
                          ,window=torch.hann_window(wl_sam)
                          ,hop_length=hl_sam
                          ,normalized=True
                          ,return_complex=False)
    
    return stft_coefs.unsqueeze(1)


def batch_istft(data: torch.tensor, 
                sr: int=8000, 
                hop_len_s: float=.016, 
                win_s: float=.064, 
                device: str='cpu'):
    """Compute the inverse short time fourier transform of a batch of audio files

    Args:
        data (torch.tensor): input complex STFT coefficients. dim: (B, 1, f_bins, h_bins, 2)
        sr (int, optional): sample rate of audio files. Defaults to 8000.
        hop_len_s (float, optional): hop length in seconds. Defaults to .016.
        win_s (float, optional): _description_. Defaults to .064.
        device (str, optional): device data is stored on. Defaults to 'cpu'.

    Returns:
        torch.tensor: real valued audio data. dim: (B, 1, segment length)
    """
    hl_sam = int(round(hop_len_s*sr))
    wl_sam = int(round(win_s*sr))

    data = data.squeeze(1)
    
    wavs = torch.istft(data
                      ,n_fft=wl_sam
                      ,window=torch.hann_window(wl_sam,device=device)
                      ,hop_length=hl_sam)
    
    return wavs.unsqueeze(1)


def split_ir_files(file_path: str, 
                   test_percent: float, 
                   val_percent: float):
    """ 
    splits impulse responses into train validation and test sets
    Takes in file path and test set percentage
    
    ln: input file path, test percent
    out: dictionary with train and test keys
    """
    files = os.listdir(file_path)
    random.seed(0)
    random.shuffle(files)
    test_size = int(len(files)*test_percent)
    val_size = int(len(files)*val_percent)
    
    test_files = files[:test_size]
    val_files = files[test_size:test_size+val_size]
    train_files = files[test_size+val_size:]
    return {'train':train_files, 'test':test_files, 'val':val_files}


def copy_ir_files(source_dir: str,
                  dest_dir: str,
                  filenames: list,
                  sr: int =8000):
    """copy impulse response files from source to destination directory

    Args:
        source_dir (str): source directory
        dest_dir (str): destination directory
        filenames (list): list of filenames to copy
        sr (int, optional): sample rate of audio files. Defaults to 8000.
    """
    for filename in os.listdir(source_dir):
        if filename in filenames:
            s_path = os.path.join(source_dir,filename)
            audio = load_wav(s_path, sr=sr)
            d_path = os.path.join(dest_dir,filename)
            write_wav(d_path, audio, sr)


def comp_out_dim(input: torch.tensor,
                 params: dict):
    
    """Computes output dimensions of a convolutional layer

    Returns:
        torch.tensor, torch.tensor: new dimensions and remainder of the division (used for calculating padding in inverse conv layers)
    """
    i = input
    k = params['k']
    s = params['s']
    p = params['p']
    
    new_dims = (i + 2*p - k)/s + 1
    new_dims_round = torch.floor(new_dims)

    rem = torch.zeros_like(new_dims_round)
    rem = tuple(((i + 2*p - k) % s).int().tolist())

    return new_dims_round, rem


def spec_plot(audio, sr=8000, n_fft=512, hop_length=128, save_png=False, png_name='test.png'):
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(abs(X)) 
    
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='jet')
    plt.colorbar(format='%+2.0f dB')
    
    if save_png:
        plt.savefig(png_name)
        
    plt.show()
    
    
def file_to_batch(wav, clip_length=32000):
    #input dim (1, file length in samples)
    #output dim (B,1, model input clip length)
    n = wav.shape[1]
    
    pad_len = clip_length - (n % clip_length)
    pad = torch.zeros((1,pad_len))
    wav = torch.cat((wav,pad),dim=1)

    wav_batch = wav.reshape(-1,1,clip_length)
    return wav_batch


def batch_to_file(batch):
    #input (B,1,clip length)
    #output (1,B*clip_length)
    wav = batch.reshape(1,-1)
    return wav

    