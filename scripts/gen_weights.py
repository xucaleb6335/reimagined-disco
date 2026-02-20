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

    w_max = np.max(np.abs(weights_float))
    i_max = np.max(np.abs(input_float))

    #cast to int8
    w8 = np.round(weights_float * (127.0/w_max)).astype(np.int8)
    i8 = np.round(input_float * (127.0/i_max)).astype(np.int8)

    hex_filename = "weights_output.hex"
    with open(hex_filename, "w") as f:
        for val in w8.flatten():
            f.write(f"{int(val) & 0xFF:02x}\n")

if __name__ == "__main__":
    main()
