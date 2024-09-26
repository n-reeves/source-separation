import torch
import torch_audiomentations as ta

import IPython.display as ipd
from sourcesep.utils import load_wav, spec_plot, batch_stft, batch_to_file, file_to_batch

def evaluate_file(filepath, model, apply_ir=True, only_tas=False, save_png=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    sr = model.sr
    in_samples = model.in_samples

    input_audio_pre_trans = load_wav(filepath, sr = sr)
    input_audio = input_audio_pre_trans
    
    if apply_ir:
        ir_trans = ta.ApplyImpulseResponse(ir_paths = './Data/val/ir'
                                                    ,p = 1
                                                    ,compensate_for_propagation_delay=True
                                                    ,sample_rate=sr)
        input_audio = ir_trans(input_audio_pre_trans.unsqueeze(0)).squeeze(0)

    #split data into batches
    input_split = file_to_batch(input_audio, in_samples, sr) #model.in_samples

    input_data = input_split

    #if evaluating joint system, input audio to stft and 
    if not only_tas:
        hop_len_s = model.hop_len_s
        win_s = model.win_s
        input_data = batch_stft(input_split, sr=sr, hop_len_s=hop_len_s, win_s=win_s)

    #model forward
    model.eval()
    with torch.no_grad():
        input_data = input_data.to(device)
        outputs = model(input_data)

    #Print Input data (pre reverb)
    print('Input')
    ipd.display(ipd.Audio(input_audio_pre_trans, rate = sr))
    spec_plot(input_audio_pre_trans.squeeze(0).detach().numpy(), save_png=save_png, png_name='input.png')

    #print Input data (post reverb)
    if apply_ir:
        print('Reverberated Input')
        ipd.display(ipd.Audio(input_audio, rate = sr))
    
    #if evaluating full system format and calc loss for intermediate outputs
    if not only_tas:
        unet_pred = outputs['m1_out']
        unet_pred = unet_pred.detach().cpu().numpy()
        unet_pred = batch_to_file(unet_pred)

        print('U-Net Out')
        ipd.display(ipd.Audio(unet_pred, rate = sr))
        spec_plot(np.squeeze(unet_pred,axis=0), save_png=save_png, png_name='UNetOut.png')
        pred_sources = outputs['m2_out']
    else:
        pred_sources = outputs

    #print predicted sources
    pred_sources = pred_sources.detach().cpu().numpy()
    s1 = pred_sources[:,0,:,:]
    s2 = pred_sources[:,1,:,:]
    
    s1 = batch_to_file(s1)
    s2 = batch_to_file(s2)

    print('TasNet Out: Source 1')
    ipd.display(ipd.Audio(s1, rate = sr))
    spec_plot(np.squeeze(s1,axis=0),save_png=save_png, png_name='Source1.png')

    print('TasNet Out: Source 2')
    ipd.display(ipd.Audio(s2, rate = sr))
    spec_plot(np.squeeze(s2,axis=0), save_png=save_png, png_name='Source2.png')

    