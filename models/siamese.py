#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:23:01 2020

@author: krishna
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.siamese = nn.Sequential(
                nn.Linear(512,512),
                nn.Linear(512,512),
                nn.Linear(512,512)
        )
        
    def forward_one(self, x):
        x = self.siamese(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1,out2
        
    

