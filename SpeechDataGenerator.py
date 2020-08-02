#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: Krishna
"""
import numpy as np
import torch
import random



def speech_collate(batch):
    input_1=[]
    input_2=[]
    output_1 = []
    output_2 = []
    labels = []
    for item in batch:
        input_1.append(item['input_1'])
        input_2.append(item['input_2'])
        output_1.append(item['output_1'])
        output_2.append(item['output_2'])
        labels.append(item['label'])
    
    return input_1,input_2, output_1,output_2,labels




class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, clean_filelink, noisy_filelink,batch_s):
        """
        Read the textfile and get the paths
        """
        self.batch_s = batch_s
        self.clean_feats = np.load(clean_filelink,allow_pickle=True).item()['features']
        self.clean_labels = np.load(clean_filelink,allow_pickle=True).item()['labels']
        
        self.noisy_feats = np.load(clean_filelink,allow_pickle=True).item()['features']
        self.noisy_labels = np.load(clean_filelink,allow_pickle=True).item()['labels']
        
        
        
    def __len__(self):
        return len(self.clean_feats)
    
    
    def __getitem__(self, idx):
        rand_index = random.sample(range(self.clean_feats.shape[0]),self.batch_s)
        get_labels_1=[]
        clean_batch_1 = []
        noisy_batch_1 = []
        for i in rand_index:
            clean_batch_1.append(self.clean_feats[i])
            get_labels_1.append(self.clean_labels[i])
            noisy_batch_1.append(self.noisy_feats[i])
            
        rand_index = random.sample(range(self.clean_feats.shape[0]),self.batch_s)
        get_labels_2=[]
        clean_batch_2 = []
        noisy_batch_2 = []
        for i in rand_index:
            clean_batch_2.append(self.clean_feats[i])
            get_labels_2.append(self.clean_labels[i])
            noisy_batch_2.append(self.noisy_feats[i])
        
        actual_label = []
        for p in range(len(get_labels_1)):
            if get_labels_1[p]==get_labels_2[p]:
                actual_label.append(1)
            else:
                actual_label.append(0)
       
        
        
        input_1 = noisy_batch_1
        input_2 = noisy_batch_2
        output_1 = clean_batch_1
        output_2 = clean_batch_2
       
        
        sample = {'input_1': torch.from_numpy(np.ascontiguousarray(input_1)), 
                  'input_2': torch.from_numpy(np.ascontiguousarray(input_2)), 
                  'output_1': torch.from_numpy(np.ascontiguousarray(output_1)), 
                  'output_2': torch.from_numpy(np.ascontiguousarray(output_2)),
                  'label': torch.from_numpy(np.ascontiguousarray(actual_label))}
        return sample
    
    
