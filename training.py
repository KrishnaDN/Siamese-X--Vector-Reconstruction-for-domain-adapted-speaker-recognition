#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:28:25 2020

@author: krishna
"""

import os
import numpy as np
from models.siamese import Siamese
import torch
from SpeechDataGenerator import SpeechDataGenerator, speech_collate
import torch.nn as nn
from torch.utils.data import DataLoader  
import argparse
import yaml
import numpy as np
from torch import optim


def train(model,dataloader_train,epoch,optimizer,device,rec_loss,cosine_loss):
    total_loss=[]
    model.train()
    print('##################### Training######################')
    for i_batch, sample_batched in enumerate(dataloader_train):
        input_1 = sample_batched[0][0]
        input_2 = sample_batched[1][0]
        output_1 = sample_batched[2][0]
        output_2 = sample_batched[3][0]
        labels = sample_batched[4][0]
        input_1,input_2,output_1,output_2 = input_1.to(device),input_2.to(device),output_1.to(device),output_2.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        out_p1,out_p2 = model(input_1,input_2)
        loss = compute_loss(rec_loss,cosine_loss,out_p1,out_p2,labels)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        
        
    print('Training Loss {} after {} epochs'.format(np.mean(np.asarray(total_loss)),epoch))
    #print('Training CER {} after {} epochs'.format(avg_cer,epoch))
    
    return np.mean(np.asarray(total_loss))

def compute_loss(rec_loss,cosine_loss,output_pred_1,ouput_pred_2,labels):
    
    rec_loss = rec_loss(output_pred_1,ouput_pred_2)
    cos_loss = cosine_loss(output_pred_1,ouput_pred_2)-labels
    final_loss = rec_loss+cos_loss
    final_loss = final_loss.mean()
    return final_loss
    

def main(config,args):
   
    
    use_cuda = config['use_gpu']
    device = torch.device("cuda" if use_cuda==1 else "cpu")
    model = Siamese()
    model = model.to(device)
    
    rec_loss = nn.L1Loss()
    cosine_loss = nn.CosineSimilarity(dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
    dataset_train = SpeechDataGenerator(args.clean_file,args.noisy_file,batch_s=100)
    dataloader_train = DataLoader(dataset_train, batch_size=1,shuffle=True,collate_fn=speech_collate) 
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss=train(model,dataloader_train,epoch,optimizer,device,rec_loss,cosine_loss)
        
if __name__ == "__main__":
    ########## Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-clean_file',type=str,default='clean.npy')
    parser.add_argument('-noisy_file',type=str, default='noisy.npy')
    parser.add_argument('-config_file',type=str, default='config.yaml')
    parser.add_argument('-save_modelpath',type=str, default='save_models/')
    
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    print(config)
    main(config,args)
    



