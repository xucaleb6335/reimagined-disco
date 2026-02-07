import tensorflow as tf
import numpy as np
import os

def main():
    tf.random.set_seed(0)
    np.random.seed(0)

    # 1. Trains a small neural network with real weights
    # 2. quantize the weights from float to int8 format
    # 3. export as a massive .hex file, so we can accelerate it on an FPGA

    weights_float = np.random.uniform(-1,1, (4,4)).astype(np.float32) #Creates a 4x4 matrix with values [-1,1]
    input_float = np.random.uniform(-1,1, (4,1)).astype(np.float32)

    #Quantization step:
    #int8_val = Round(Float_Value*127.0 / Max_Abs_value)

    w_max = np.max(weights_float)
        
