import random
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

# Cocotb testbench for accelerator_top
# Replays every .npz vector from scripts/golden_model.py through the
# AXI-Stream sink (with randomized tvalid stalls for backpressure), kicks
# off a run via the Avalon-MM CSR, polls DONE and checks the 32 bit
# accumulators against the NumPy golden product

DATA_DIR = Path(__file__).resolve().parent / "data"

CTRL_ADDR   = 0
STATUS_ADDR = 1
RESULT_BASE = 2

DONE_TIMEOUT = 200 #poll iterations before declaring the run hung


def to_signed32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


def pack_lanes(vec):
    #lane n of tdata is tdata[(n+1)*8-1 -: 8]
    word = 0
    for n, v in enumerate(vec):
        word |= (int(v) & 0xFF) << (8 * n)
    return word


def matrix_to_beats(a, b):
    #Beats 0..N-1: matrix A, one column per beat (lane i = A[i][k])
    #Beats N..2N-1: matrix B, one row per beat   (lane j = B[k][j])
    size = a.shape[0]
    beats = [pack_lanes(a[:, k]) for k in range(size)]
    beats += [pack_lanes(b[k, :]) for k in range(size)]
    return beats


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.tvalid.value = 0
    dut.tdata.value = 0
    dut.avs_address.value = 0
    dut.avs_write.value = 0
    dut.avs_read.value = 0
    dut.avs_writedata.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)


async def axis_send(dut, beats, stall_prob):
    #Randomized tvalid gaps simulate a stalling bus master; tready
    #backpressure from the DUT is honoured by sampling before the edge
    for beat in beats:
        while random.random() < stall_prob:
            dut.tvalid.value = 0
            await RisingEdge(dut.clk)
        dut.tvalid.value = 1
        dut.tdata.value = beat
        while True:
            await ReadOnly()
            ready = bool(dut.tready.value)
            await RisingEdge(dut.clk)
            if ready:
                break
    dut.tvalid.value = 0


async def avalon_write(dut, addr, data):
    dut.avs_address.value = addr
    dut.avs_writedata.value = data
    dut.avs_write.value = 1
    await RisingEdge(dut.clk)
    dut.avs_write.value = 0


async def avalon_read(dut, addr):
    #Fixed 1 cycle read latency (registered readdata in avalon_csr)
    dut.avs_address.value = addr
    dut.avs_read.value = 1
    await RisingEdge(dut.clk)
    dut.avs_read.value = 0
    await ReadOnly()
    value = int(dut.avs_readdata.value)
    await RisingEdge(dut.clk)
    return value


async def run_matmul(dut, a, b, c_gold, name, stall_prob):
    size = a.shape[0]

    await axis_send(dut, matrix_to_beats(a, b), stall_prob)
    await avalon_write(dut, CTRL_ADDR, 1) #START

    for _ in range(DONE_TIMEOUT):
        if await avalon_read(dut, STATUS_ADDR) & 1:
            break
    else:
        raise AssertionError(f"{name}: DONE never asserted")

    #Result window is C[i][j] row-major from RESULT_BASE
    c_dut = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            raw = await avalon_read(dut, RESULT_BASE + i * size + j)
            c_dut[i][j] = to_signed32(raw)

    assert (c_dut == c_gold).all(), (
        f"{name}: mismatch\nA=\n{a}\nB=\n{b}\nDUT=\n{c_dut}\nGOLD=\n{c_gold}"
    )
    dut._log.info(f"{name}: PASS")


@cocotb.test()
async def accelerator_golden_cases(dut):
    """Every golden_model.py vector, moderate random backpressure"""
    random.seed(0)
    cases = sorted(DATA_DIR.glob("*.npz"))
    assert cases, f"no test vectors in {DATA_DIR}; run scripts/golden_model.py"

    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    await reset_dut(dut)

    #Back-to-back runs without reset also prove clear_acc works
    for case in cases:
        data = np.load(case)
        await run_matmul(dut, data["a"], data["b"], data["c"], case.stem,
                         stall_prob=0.3)


@cocotb.test()
async def accelerator_heavy_backpressure(dut):
    """One vector streamed through a heavily stalling bus"""
    random.seed(1)
    cases = sorted(DATA_DIR.glob("*.npz"))
    assert cases, f"no test vectors in {DATA_DIR}; run scripts/golden_model.py"

    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    await reset_dut(dut)

    data = np.load(cases[0])
    await run_matmul(dut, data["a"], data["b"], data["c"], cases[0].stem,
                     stall_prob=0.8)
