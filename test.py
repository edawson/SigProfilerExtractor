#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:45:38 2019

@author: mishugeb
"""
from SigProfilerExtractor import sigpro as sig
import torch

def test_cpu(data):
    sig.sigProfilerExtractor("text", "example_output", data, startProcess=1, endProcess=2, totalIterations=3)

def test_gpu(data):
    if torch.cuda.device_count() > 0:
        sig.sigProfilerExtractor("text", "example_output", data, startProcess=1, endProcess=2, totalIterations=3, gpu=True)

def main():
    data = sig.importdata("text")
    test_cpu(data)
    test_gpu(data)

if __name__ == '__main__':
    main()
