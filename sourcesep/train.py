import torch
import IPython.display as ipd

import random

def evaluate(model, 
             data_loader, 
             criterion, 
             device, 
             batch_eval=500, 
             fine_tune=False):
    model.eval()
    sep_net = model.sep_net 
    num_batches = len(data_loader)
    epoch_loss = 0.
    epoch_si_snri = 0.
    epoch_si_snri_dn = 0.
    print('Starting validation')
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inp_signal = batch['signal_mix_input'] 
            inp_stft = batch['stft_mix_input']
            m1_lab_clean = batch['signal_mix_clean']
            m1_lab_dereverb = batch['signal_mix_noise']
            
            inp_stft = inp_stft.to(device)
            inp_signal = inp_signal.to(device)
            m1_lab_clean = m1_lab_clean.to(device)
            m1_lab_dereverb = m1_lab_dereverb.to(device)

            pred_dict = model(inp_stft)
            m1_pred = pred_dict['m1_out']

            #loss for predicted denoised and deverbed signal
            m1_loss = criterion(m1_pred, m1_lab_clean)

            #for debugging
            if torch.isnan(m1_loss).any():
                print('lab dim')
                print(m1_lab_clean.shape)
                print('input audio')
                
                inp_signal = inp_signal.detach().cpu().numpy()
                m1_pred = m1_pred.detach().cpu().numpy()
                batch_size = m1_pred.shape[0]
                for i in range(batch_size):
                    print('item {} preds: clean and deverb '.format(i))
                    ipd.display(ipd.Audio(inp_signal[i] ,rate=model.sr))
                    ipd.display(ipd.Audio(m1_pred[i] ,rate=model.sr))

            #if tasnet is present in source separation network, calc loss
            if sep_net:
                m2_lab = batch['sources']
                m2_lab = m2_lab.to(device)
                
                m2_preds = pred_dict['m2_out']
                m2_loss = criterion(m2_preds,m2_lab)

            #if fine tuning, only use source seperation  loss
            #otherwise if the seperation net is active, sum the losses
            #last case is U-net only training, only use the enhacnemnt loss 
            if fine_tune:
                loss = m2_loss
            elif sep_net:
                loss = m1_loss + m2_loss
            else:
                loss = m1_loss

            batch_loss = loss.item()
            epoch_loss += batch_loss

            #calculate si snri for the denoiser
            #(was not run during training) added after for test evaluation
            base_loss_dn = criterion(inp_signal, m1_lab_clean)
            base_si_snr_dn =  base_loss_dn.item() * -1
            pred_si_snr_dn = m1_loss.item() * -1
            batch_si_snri_dn = pred_si_snr_dn - base_si_snr_dn
            epoch_si_snri_dn += batch_si_snri_dn

            #if tasnet is present in model, calculate imrovement in signal to noise ratio over original audio
            batch_si_snri = 0.
            if sep_net:
                inp_eval = inp_signal.unsqueeze(1).repeat(1,2,1,1)
                base_loss = criterion(inp_eval, m2_lab)
                base_si_snr = base_loss.item() * -1
                pred_si_snr = m2_loss.item() * -1
                
                batch_si_snri = pred_si_snr - base_si_snr
                epoch_si_snri += batch_si_snri

            if i % batch_eval == 0:
                print('Batch {}'.format(i+1))
                print('Batch U-Net Loss: {}'.format(m1_loss.item()))
                print('Batch Total Loss: {}'.format(batch_loss))
                print('Batch SI-SNRi: {}'.format(batch_si_snri))
            
    epoch_loss /= num_batches
    epoch_si_snri /= num_batches
    epoch_si_snri_dn /= num_batches

    #display random utter from last example
    inp_signal = inp_signal.detach().cpu().numpy()
    m1_lab_clean = m1_lab_clean.squeeze(1).detach().cpu().numpy()
    m1_lab_dereverb = m1_lab_dereverb.squeeze(1).detach().cpu().numpy()
    
    m1_pred = m1_pred.squeeze(1).detach().cpu().numpy()
    
    ind = random.randrange(inp_signal.shape[0])
    sr = model.sr

    #display audio
    print('Input Audio')
    ipd.display(ipd.Audio(inp_signal[ind],rate=sr))

    #Clean mask
    print('Target Clean mixture')
    ipd.display(ipd.Audio(m1_lab_clean[ind],rate=sr))

    #Dereverb mask
    print('Target Noisy mixture')
    ipd.display(ipd.Audio(m1_lab_dereverb[ind],rate=sr))


    print('Predicted Unet mixture')
    ipd.display(ipd.Audio(m1_pred[ind],rate=sr))

    if sep_net:
        out_ex = m2_preds[ind]
        output = out_ex.permute(1,0,2).squeeze(2).reshape((2,-1))
    
        pred_s1 = output[0,:].cpu().detach().numpy()
        pred_s2 = output[1,:].cpu().detach().numpy()
    
        src_ex = m2_lab[ind]
        lab_s1 = src_ex[0,:].cpu().detach().numpy()
        lab_s2 = src_ex[1,:].cpu().detach().numpy()

        print('Predicted Sources')
        ipd.display(ipd.Audio(pred_s1,rate=sr))
        ipd.display(ipd.Audio(pred_s2,rate=sr))
    
        print('True Sources')
        ipd.display(ipd.Audio(lab_s1,rate=sr))
        ipd.display(ipd.Audio(lab_s2,rate=sr))

    return epoch_loss, epoch_si_snri, epoch_si_snri_dn



def train(epochs
          ,model 
          ,criterion
          ,optimizer
          ,scheduler
          ,train_loader
          ,test_loader
          ,device
          ,save_dir='./models/test.pkl'
          ,valid_freq=10
          ,fine_tune=False
          ,batch_eval=500
          ,batch_loud=True):
    
    model.train()
    num_batches = len(train_loader)
    best_si_snri = 0.
    best_valid_loss = 0.
    train_losses = []
    train_si_snris = []
    valid_losses = []
    valid_si_snris = []
    sep_net = model.sep_net

    print('{} batches per epoch'.format(num_batches))
    
    for epoch in range(epochs):
        print('starting epoch: {}'.format(epoch))
        model.train()
        epoch_loss = 0.
        epoch_m1_loss = 0.
        epoch_si_snri = 0.
        for i, batch in enumerate(train_loader):
            inp_signal = batch['signal_mix_input']
            inp_stft = batch['stft_mix_input']
            m1_lab_clean = batch['signal_mix_clean']
            m1_lab_dereverb = batch['signal_mix_noise']
            
            inp_stft = inp_stft.to(device)
            inp_signal = inp_signal.to(device)
            m1_lab_clean = m1_lab_clean.to(device)
            m1_lab_dereverb = m1_lab_dereverb.to(device)

            pred_dict = model(inp_stft)
            m1_pred = pred_dict['m1_out']

            #loss for predicted denoised and deverbed signal
            m1_loss = criterion(m1_pred, m1_lab_clean)

            #debugging
            if torch.isnan(m1_loss).any():
                print('lab dim')
                print(m1_lab_clean.shape)
                print('input audio')
                
                inp_signal = inp_signal.detach().cpu().numpy()
                m1_pred = m1_pred.detach().cpu().numpy()
                batch_size = m1_pred.shape[0]
                for i in range(batch_size):
                    print('item {} preds: clean and deverb '.format(i))
                    ipd.display(ipd.Audio(inp_signal[i] ,rate=model.sr))
                    ipd.display(ipd.Audio(m1_pred[i] ,rate=model.sr))

            #if seperation net is being trained
            if sep_net:
                m2_lab = batch['sources']
                m2_lab = m2_lab.to(device)
                
                m2_preds = pred_dict['m2_out']
                m2_loss = criterion(m2_preds, m2_lab)

            
            if fine_tune: #if fine tuning, loss is the source separation loss
                loss = m2_loss
            elif sep_net: #if training in the first phase, both unet and tasnet loss are used
                loss = m1_loss + m2_loss
            else:  #if only training the unet, loss is 
                loss = m1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_m1_loss += m1_loss.item()

            #if evaluating second model, calculate improvement in si-snr over the noisy mixture
            batch_si_snri = 0.
            if sep_net:
                inp_eval = inp_signal.unsqueeze(1).repeat(1,2,1,1)
                base_loss = criterion(inp_eval, m2_lab)
                base_si_snr = base_loss.item() * -1
                pred_si_snr = m2_loss.item() * -1
                
                batch_si_snri = pred_si_snr - base_si_snr
                epoch_si_snri += batch_si_snri

            if i % batch_eval == 0:
                print('Batch {}'.format(i+1))
                print('Batch U-Net Loss: {}'.format(m1_loss.item()))
                print('Batch Total Loss: {}'.format(batch_loss))
                print('Batch SI-SNRi: {}'.format(batch_si_snri))

                if batch_loud:
                    inp_signal = inp_signal.detach().cpu().numpy()
                    m1_pred = m1_pred.detach().cpu().numpy()
                    batch_size = m1_pred.shape[0]
                    
                    for i in range(batch_size):
                        print('item {} preds: clean and deverb '.format(i))
                        ipd.display(ipd.Audio(inp_signal[i] ,rate=model.sr))
                        ipd.display(ipd.Audio(m1_pred[i] ,rate=model.sr))

        epoch_loss /= num_batches
        epoch_m1_loss /= num_batches
        epoch_si_snri /= num_batches
        
        print(f'[{epoch+1}] loss: {epoch_loss:.6f}')
        print(f'[{epoch+1}] UNet loss: {epoch_m1_loss:.6f}')
        
        print(f'[{epoch+1}] si-snri: {epoch_si_snri:.6f}')
        train_losses.append(epoch_loss)
        train_si_snris.append(epoch_si_snri)

        #halve rate if loss doesn't decrease after one epochs
        scheduler.step(epoch_loss)
        
        if((epoch+1) % valid_freq == 0):
                valid_loss, valid_si_snri, valid_si_snri_dn = evaluate(model, test_loader, criterion, device, batch_eval, fine_tune)
                print(f'Validation loss: {valid_loss:.6f}')
                print(f'SI-SNRi: {valid_si_snri:.6f}')
                valid_losses.append(valid_loss)
                valid_si_snris.append(valid_si_snri)
                
                #When only training denoiser, save model if it minimizes loss
                #if training separator, save model when valid si_snri improves
                if(sep_net and valid_si_snri >= best_si_snri) or (not sep_net and best_valid_loss > valid_loss):
                    best_si_snri = valid_si_snri
                    best_valid_loss = valid_loss
                    print('Saving best model')
                    torch.save(model.state_dict(), save_dir)
    return train_losses, valid_losses, train_si_snris, valid_si_snris


#evaluation functions for source separation network only (not joint)
#should be removed in the future due to redundency

def evaluate_ss(model, data_loader, criterion, device):
    model.eval()
    num_batches = len(data_loader)
    epoch_loss = 0.
    si_snri = 0.
    
    with torch.no_grad():
        for batch in data_loader:
            inp = batch['signal_mix_input']
            lab = batch['sources']
            
            inp = inp.to(device)
            lab = lab.to(device)
            
            preds = model(inp)
            loss = criterion(preds, lab)
            pred_loss = loss.item()
            epoch_loss += pred_loss

            pred_sisnr = pred_loss*-1

            #calc SI-NRI: ratio of SI NRI for pred sources to original mixture
            inp_eval = inp.unsqueeze(1).repeat(1,2,1,1)
            unalt_loss = criterion(inp_eval, lab)
            mix_sisnr = unalt_loss.item() * -1
            
            si_snri += pred_sisnr - mix_sisnr
            
    epoch_loss /= num_batches
    si_snri /= num_batches

    #display random utter from last example
    inp = inp.detach().cpu().numpy()
    ind = random.randrange(inp.shape[0])
    sr = model.sr
    
    out_ex = preds[ind]
    output = out_ex.permute(1,0,2).squeeze(2).reshape((2,-1))

    pred_s1 = output[0,:].cpu().detach().numpy()
    pred_s2 = output[1,:].cpu().detach().numpy()

    lab_ex = lab[ind]
    lab_s1 = lab_ex[0,:].cpu().detach().numpy()
    lab_s2 = lab_ex[1,:].cpu().detach().numpy()

    #display audio
    print('Input Audio')
    ipd.display(ipd.Audio(inp[ind],rate=sr))

    print('Predicted Sources')
    ipd.display(ipd.Audio(pred_s1,rate=sr))
    ipd.display(ipd.Audio(pred_s2,rate=sr))

    print('True Sources')
    ipd.display(ipd.Audio(lab_s1,rate=sr))
    ipd.display(ipd.Audio(lab_s2,rate=sr))

    
    return epoch_loss, si_snri


def train_ss(epochs
          ,model 
          ,criterion
          ,optimizer
          ,scheduler
          ,train_loader
          ,test_loader
          ,device
          ,batch_eval=500
          ,save_dir='./models/test.pkl'
          ,valid_freq=10):
    model.train()
    num_batches = len(train_loader)
    best_si_snri = 0.
    train_losses = []
    train_si_snri = []
    valid_losses = []
    valid_si_snri = []
    
    for epoch in range(epochs):
        print('starting epoch: {}'.format(epoch))
        model.train()
        epoch_loss = 0.
        epoch_si_snri = 0.
        print('Batched per epoch: {}'.format(len(train_loader)))
        for i,batch in enumerate(train_loader):
            inp = batch['signal_mix_input']
            lab = batch['sources']
            
            inp = inp.to(device)
            lab = lab.to(device)
            preds = model(inp)
            loss = criterion(preds,lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss

            #calc si-snri
            pred_sisnr = -1*batch_loss
            inp_eval = inp.unsqueeze(1).repeat(1,2,1,1)
            unalt_loss = criterion(inp_eval, lab)
            mix_sisnr = unalt_loss.item() * -1
            batch_si_snri = pred_sisnr - mix_sisnr
            epoch_si_snri += batch_si_snri

            if i % batch_eval == 0:
                print('Batch {}'.format(i+1))
                print('Batch Loss: {}'.format(batch_loss))
                print('Batch SI-SNRi: {}'.format(batch_si_snri))

        epoch_loss /= num_batches
        epoch_si_snri /= num_batches
        
        print(f'[{epoch+1}] loss: {epoch_loss:.6f}')
        print(f'[{epoch+1}] si-snri: {epoch_si_snri:.6f}')
        train_losses.append(epoch_loss)
        train_si_snri.append(epoch_si_snri)

        #halve learning rate if plateau
        scheduler.step(epoch_loss)
        
        if((epoch+1) % valid_freq == 0):
                valid_loss, si_snri = evaluate_ss(model, test_loader, criterion, device)
                print(f'Validation loss: {valid_loss:.6f}')
                print(f'SI-SNRi: {si_snri:.6f}')
                valid_losses.append(valid_loss)
                valid_si_snri.append(si_snri)
                
                # if the best validation performance so far, save the network to file 
                if(si_snri >= best_si_snri):
                    best_si_snri = si_snri
                    print('Saving best model')
                    torch.save(model.state_dict(), save_dir)
    return train_losses, valid_losses, train_si_snri, valid_si_snri