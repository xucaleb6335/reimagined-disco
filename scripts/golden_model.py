import numpy as np
import os

# Golden reference generator for the int8 systolic array accelerator
# 1. builds int8 A/B matrices for each edge case
# 2. computes the int32 golden product C = A x B with NumPy
# 3. exports .npz vectors into verif/cocotb/data/ for tb_accelerator.py

ARRAY_SIZE = 4
NUM_RANDOM = 5

def write_case(out_dir, name, a, b):
    a = a.astype(np.int8)
    b = b.astype(np.int8)
    #int32 accumulators; matches the 4*DATA_WIDTH acc in pe.sv
    c = a.astype(np.int32) @ b.astype(np.int32)
    np.savez(os.path.join(out_dir, f"{name}.npz"), a=a, b=b, c=c)
    print(f"wrote {name}.npz (max |c| = {np.max(np.abs(c))})")

def main():
    np.random.seed(0)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "verif", "cocotb", "data")
    os.makedirs(out_dir, exist_ok=True)

    #Standard randomized
    for n in range(NUM_RANDOM):
        a = np.random.randint(-128, 128, (ARRAY_SIZE, ARRAY_SIZE))
        b = np.random.randint(-128, 128, (ARRAY_SIZE, ARRAY_SIZE))
        write_case(out_dir, f"random_{n}", a, b)

    #Identity: C = I x B = B
    b = np.random.randint(-128, 128, (ARRAY_SIZE, ARRAY_SIZE))
    write_case(out_dir, "identity", np.eye(ARRAY_SIZE), b)

    #All zeros: C = 0
    a = np.random.randint(-128, 128, (ARRAY_SIZE, ARRAY_SIZE))
    write_case(out_dir, "zeros", a, np.zeros((ARRAY_SIZE, ARRAY_SIZE)))

    #Sparse: half of both operands zeroed out
    a = np.random.randint(-128, 128, (ARRAY_SIZE, ARRAY_SIZE))
    b = np.random.randint(-128, 128, (ARRAY_SIZE, ARRAY_SIZE))
    a[np.random.rand(ARRAY_SIZE, ARRAY_SIZE) < 0.5] = 0
    b[np.random.rand(ARRAY_SIZE, ARRAY_SIZE) < 0.5] = 0
    write_case(out_dir, "sparse", a, b)

    #Saturation bounds: worst-case accumulator magnitudes
    write_case(out_dir, "saturation_min",
               np.full((ARRAY_SIZE, ARRAY_SIZE), -128),
               np.full((ARRAY_SIZE, ARRAY_SIZE), -128))
    write_case(out_dir, "saturation_mix",
               np.full((ARRAY_SIZE, ARRAY_SIZE), -128),
               np.full((ARRAY_SIZE, ARRAY_SIZE), 127))

if __name__ == "__main__":
    main()
