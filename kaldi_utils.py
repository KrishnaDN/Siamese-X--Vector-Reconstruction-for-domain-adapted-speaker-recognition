#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:00:49 2020

@author: krishna
"""

import os
import numpy as np
import glob
from shutil import rmtree
import kaldi_io


class Dataset(object):
    def __init__(self, audio_folder,kaldi_dir,data_type):
        super(Dataset, self).__init__()
        self.audio_folder = audio_folder
        self.data_type = data_type
        self.kaldi_dir = kaldi_dir
        self.cur_dir = os.getcwd()
        
    def get_speakers(self):
        speaker_folders = sorted(glob.glob(self.audio_folder+'/*/'))
        speaker_names = [speaker.split('/')[-2] for speaker in speaker_folders]
        speaker_ids ={}
        for i in range(len(speaker_names)):
            speaker_ids[speaker_names[i]]=i
        return speaker_ids

    def create_kaldi(self):
        if not os.path.exists('data/test'):
            os.makedirs('data/test')
        wav_scp = open('data/test/wav.scp','w')
        utt2spk = open('data/test/utt2spk','w')
        spk2utt = open('data/test/spk2utt','w')
        speaker_folders = sorted(glob.glob(self.audio_folder+'/*/'))
        for speaker in speaker_folders:
            files = sorted(glob.glob(speaker+'/*.wav'))
            for filepath in files:
                to_wav_scp = filepath.split('/')[-1]+' '+filepath
                to_utt2spk = filepath.split('/')[-1]+' '+filepath.split('/')[-1]
                wav_scp.write(to_wav_scp+'\n')
                utt2spk.write(to_utt2spk+'\n')
                spk2utt.write(to_utt2spk+'\n')
        wav_scp.close()
        utt2spk.close()
        spk2utt.close()
        
    
    def clean_up(self):
        rmtree('data/test/')
    
    
    
    def read_features(self):
        features_folder = 'exp/xvector_nnet_1a/xvectors_test/'
        all_scps = sorted(glob.glob(self.kaldi_dir+'/'+features_folder+'/xvector.*.scp'))
        speaker_map = self.get_speakers()
        all_features= []
        all_labels=[]
        for scp_file in all_scps:
            for key,mat in kaldi_io.read_vec_flt_scp(scp_file):
                speaker_name = speaker_map[key[:-13]]
                print(key)
                all_features.append(mat)
                all_labels.append(speaker_name)
        return np.asarray(all_features), np.asarray(all_labels)
        
    
    def call_extractor(self):
        os.chdir(self.kaldi_dir)
        self.create_kaldi()
        os.system('sh extractor.sh')
        features,labels = self.read_features()
        save_data = {}
        save_data['features'] = features
        save_data['labels'] = labels
        save_path = self.cur_dir+'/'+self.data_type+'.npy'
        self.clean_up()
        os.chdir(self.cur_dir)
        np.save(save_path,save_data)
        
   

if __name__=='__main__':
    
    audio_folder = '/home/krishna/Krishna/paper_implementations/Siamese_x_vector/Dataset/clean'
    kaldi_dir = '/home/krishna/Krishna/kaldi/egs/voxceleb/v3'
    data_type = 'clean'
    dataset = Dataset(audio_folder,kaldi_dir,data_type)
    dataset.call_extractor()
    





    
    
    
    
    
    
    
    
    
    