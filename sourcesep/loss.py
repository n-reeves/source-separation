import torch

# Used in old data augmentation method zero padding is added to end of clip and evenly split
def get_pad_mask(labels:torch.tensor):
    """
    Add padding to the end of each utterance in a batch of labels and return a mask to remove the padding from the loss calculation
    
    Args:
        labels (torch.tensor): tensor of clean speaker signals with shape (batch, samples)

    Returns:
        torch.tensor: mask to remove padded values from loss calculation
    """
    
    #input :(B,N)
    #output: (B,N)
    #producing mask corresponding to zero padding added to an utterance
    sources = labels.shape[0]
    samples = labels.shape[1]
    
    labels_fl = torch.flip(labels, dims=[1])
    
    #returns the first non-zero index for the flipped matrix
    first_non_zero_fl = (labels_fl != 0).type(torch.float64).argmax(dim=1, keepdim=True)
    
    #constructs ranges of indices for each source and sample
    mask_range = torch.arange(0,samples).repeat(sources, 1).to(labels.device)

    #creates matrix of zeros and ones. zeros correspond to stretches of zero values at the end of each utterance
    mask_fl = (mask_range >= first_non_zero_fl).type(torch.float64)
    mask = torch.flip(mask_fl, dims=[1])
    return mask


def si_snr(preds: torch.tensor,
           labels:torch.tensor,
           mask: bool=False): 
    """
    Calculate the scale invariant source to noise ratio for a batch of predictions and labels
        case 1: labels are clean speaker mixture signals
        case 2: labels are individual speaker signals
    Args:
        preds (torch.tensor): predicted speaker signals with shape (batch,num_sources,1,clip_len) or (batch,1,clip_len)
        labels (torch.tensor): clean speaker signals
        mask (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    
    #in dim: (batch,num_sources,1,clip_len) or (batch,1,clip_len)
    #out: (1,) mean of batch losses
    
    #calculates the scale invariant source to noise ratio
    #perm=True specifies permutation invariant loss used for two speaker source seperation objective
    #note: Not generalized to S>2, implimentation of permutation invariance would need to be expanded
    
    #add num_sources dims for 
    if len(preds.shape) < 4:
        preds = preds.unsqueeze(1)

    if len(labels.shape) < 4:
        labels = labels.unsqueeze(1)

    batch_size = preds.shape[0]
    sources = preds.shape[1]
    
    preds = preds.squeeze(2).reshape(batch_size*sources, -1)
    labels = labels.squeeze(2).reshape(batch_size*sources, -1)

    if mask:
        #get mask corresponding to strings of zeros added to end of utterances 
        mask = get_pad_mask(labels)
        #apply mask to remove predictions for padded input
        preds = preds*mask

        #calc average of non-padded samples
        pred_mu = torch.sum(preds, dim=1, keepdim=True)/torch.sum(mask, dim=1, keepdim=True)
        label_mu = torch.sum(labels, dim=1, keepdim=True)/torch.sum(mask, dim=1, keepdim=True)

        #normalize
        pred_norm = preds - pred_mu
        label_norm = labels - label_mu

        #Apply mask after norm to remove non-zero values in padding
        pred_norm = pred_norm*mask
        label_norm = label_norm*mask
        
    else:
        #zero mean normalization
        pred_mu = torch.mean(preds, dim=1, keepdim=True)
        label_mu = torch.mean(labels, dim=1, keepdim=True)

        pred_norm = preds - pred_mu
        label_norm = labels - label_mu
        
    #Compute scale invariant source to noise ratio
    #consider all permutations of predictions and GT sources. If only one permutation is considered the model
    #is penalized if it gets the order of sources in regards to the labels incorrect
    #model could perform well, but order the sources in a way that doesn't agree with the ordering of the sources in the data
    #Therefore, we need to calc the loss for all permutations and take the loss that maximizes the objective (min of loss out)

    #reshape to dim that differ in 2,3 column so that preds_norm*label_norm has dim (batches, num_sources,num_sources,num_samples)
    #preds_norm * label_norm are all permutations of the elementwise products of true sources (labels),
    #and the source predictions (preds)
    #for each prediction (dim 1),dim 2 contains element wise product of prediction and each source
    #this method reshaping applies to cases when permutation invariance is not required as well
    
    pred_norm = pred_norm.reshape(batch_size, sources, 1, -1) #s_estimate
    label_norm = label_norm.reshape(batch_size, 1, sources, -1) #s_target

    #dot product of each pred,source permutation
    #dim (batch,num_sources,num_sources,1 )
    sp_scale = torch.sum(pred_norm*label_norm, dim=3, keepdim=True) 

    #signal power of each source
    #dim (batch,1,num_sources,1 )
    sp_lab = torch.sum(label_norm*label_norm, dim=3, keepdim=True) + 1e-8 #constant to avoid divide by zero errors

    #compute s target, term helps account for differences in scale across signals
    #each source signal is multiplied by the 2 possible permutations of dot products 
    #and divided by its signal power. out dim (batch,2,2, clip length)
    s_target = sp_scale*label_norm/sp_lab

    #signal difference between each prediction and each source
    #out dim: (batch, num_sources, num_sources,1)
    error = pred_norm - s_target
    
    #calculate dot product of each permuatations error and source
    s_tar_sp = torch.sum(s_target*s_target, dim=3, keepdim=True) + 1e-8 
    error_sp = torch.sum(error*error, dim=3, keepdim=True) + 1e-8 

    #calc objectives for each permutation
    #out dim: (batch,num_sources,num_sources ,1)
    si_snr = 10 *(torch.log10(s_tar_sp) - torch.log10(error_sp)) 

    #now we have a set of S objective values for each sample in the batch
    #the representation is a SxS matrix. 

    if sources == 2:
        #For the two source case:
        #the value in ind 0,0 is the objective value if the first predicted source is paired with the first target source
        #the snr in ind 0,1 is the snr for the first predicted source paired with the second target source
        #expanding on this, we can see that the permuatation that maximizes the objective for some b in batch B is either
        #si_snr[b,0,0] + si_snr[b,1,1] or si_snr[b,0,1] + si_snr[b,1,0]
    
        #dim (b, 1, num_sources,num_sources) each nsxns sub array is a matrix of objective values for each perm of sources and preds
        si_snr = si_snr.permute(0,3,1,2)
        
        #create and apply masks for each possible pairing of preds to labels
        perm_mask1 = torch.tensor([[[1,0],[0,1]]]).to(preds.device)
        perm_mask2 = torch.tensor([[[0,1],[1,0]]]).to(preds.device)
        si_snr_p1 = si_snr*perm_mask1
        si_snr_p2 = si_snr*perm_mask2
    
        #dim (b,num_sources,num_sources,num_sources)
        si_snr_p = torch.cat((si_snr_p1,si_snr_p2),dim=1)
    
        #sum objectives for each permutation and take the max to find best scoring pairs
        si_snr_p = torch.sum(si_snr_p, dim=(2,3))
        si_snr_max = torch.max(si_snr_p, dim=1, keepdim=True)[0]
        si_snr_max = si_snr_max/2 #loss is the mean of the si_snr for both sources in the permutation that maximizes objective
        
        #take mean of losses from batch and multiply by -1
        loss = -1*torch.mean(si_snr_max)

    else:
        #for the one label case, dim (b,1,1,1) -> (b,1)
        si_snr = si_snr.reshape(batch_size, 1)
        loss = -1*torch.mean(si_snr)
    
    return loss