import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

# Cocotb test for the single PE: 100 random signed MACs against a
# python reference accumulator, then a clear_acc check


def to_signed(v, bits):
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


@cocotb.test()
async def pe_random_mac(dut):
    """100 random int8 MACs vs a reference accumulator"""
    random.seed(0)
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())

    dut.rst_n.value = 0
    dut.enable.value = 0
    dut.clear_acc.value = 0
    dut.in_a.value = 0
    dut.in_b.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    dut.enable.value = 1

    expected_acc = 0
    for _ in range(100):
        a = random.randint(-128, 127)
        b = random.randint(-128, 127)
        dut.in_a.value = a & 0xFF
        dut.in_b.value = b & 0xFF
        await RisingEdge(dut.clk) #DUT accumulates these operands at this edge
        expected_acc += a * b

    dut.enable.value = 0 #Freeze the accumulator
    await ReadOnly()
    acc = to_signed(int(dut.acc.value), 32)
    assert acc == expected_acc, f"DUT={acc} Ref={expected_acc}"

    #clear_acc wipes the accumulator without a reset
    await RisingEdge(dut.clk)
    dut.clear_acc.value = 1
    await RisingEdge(dut.clk)
    dut.clear_acc.value = 0
    await ReadOnly()
    assert int(dut.acc.value) == 0, "clear_acc failed"
