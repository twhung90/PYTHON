# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:19:47 2025

@author: survey
"""

import torch
import time

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define tensor sizes to test
tensor_sizes = [(32, 64, 64), (32, 128, 128), (32, 256, 256), (32, 512, 512)]

# Function to measure computation time for a given tensor size
def measure_time(tensor_size):
    # Create a tensor on CPU
    a_cpu = torch.normal(mean=0, std=1, size=tensor_size)
    # Move tensor to GPU if available
    a_gpu = a_cpu.to(device)

    # Measure CPU time
    start_time_cpu = time.time()
    for _ in range(100000):
        b_cpu = a_cpu ** 2
    cpu_time = time.time() - start_time_cpu

    # Measure GPU time
    start_time_gpu = time.time()
    for _ in range(100000):
        b_gpu = a_gpu ** 2
    gpu_time = time.time() - start_time_gpu

    return cpu_time, gpu_time

# Run tests and print results
for size in tensor_sizes:
    cpu_time, gpu_time = measure_time(size)
    print(f'Tensor size: {size} | Average CPU Time: {cpu_time:.5f} s | Average GPU Time: {gpu_time:.5f} s')
