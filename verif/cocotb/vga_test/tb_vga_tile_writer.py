import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

# Cocotb testbench for vga_tile_writer in isolation (plan item B.1):
# drive a known c_out, release tiles with frame_tick, and check every
# accepted Avalon-MM write (address, RGB565 data, order) against a
# golden pixel list computed in Python — with random waitrequest stalls

# Mirror of the RTL parameters (defaults in vga_tile_writer.sv)
ARRAY_SIZE  = 4
TILE_SIZE   = 16
TILE_GAP    = 2
ORIGIN_X    = 8
ORIGIN_Y    = 8
Y_SHIFT     = 10
BASE_ADDR   = 0
COLOR_SHIFT = 11

TILE_PITCH = TILE_SIZE + TILE_GAP
ACC_WIDTH  = 32

#One value per tile: zero, small/large positives (incl. saturation),
#negatives, and both int32 extremes
TEST_MATRIX = [
    [0,           1,        2047,       2048],
    [4096,        65024,    2**31 - 1,  -1],
    [-2047,       -2048,    -65024,     -(2**31)],
    [123456,      -123456,  31 << 11,   -(31 << 11)],
]


def colormap(acc):
    #Keep in sync with the colormap function in vga_tile_writer.sv
    if acc == 0:
        return 0
    mag = abs(acc) >> COLOR_SHIFT
    mag5 = min(mag, 31)
    return mag5 if acc < 0 else mag5 << 11


def golden_tile(i, j, acc):
    #Write order is px fastest, then py, matching the RTL counters
    color = colormap(acc)
    pixels = []
    for py in range(TILE_SIZE):
        for px in range(TILE_SIZE):
            x = ORIGIN_X + j * TILE_PITCH + px
            y = ORIGIN_Y + i * TILE_PITCH + py
            addr = BASE_ADDR + (y << Y_SHIFT) + (x << 1)
            pixels.append((addr, color))
    return pixels


def pack_c_flat(matrix):
    #c_flat holds c[i][j] at bits [(i*ARRAY_SIZE+j+1)*32-1 -: 32]
    word = 0
    for i in range(ARRAY_SIZE):
        for j in range(ARRAY_SIZE):
            v = matrix[i][j] & 0xFFFF_FFFF
            word |= v << ((i * ARRAY_SIZE + j) * ACC_WIDTH)
    return word


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.enable.value = 0
    dut.frame_tick.value = 0
    dut.c_flat.value = 0
    dut.avm_waitrequest.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)


async def waitrequest_driver(dut, stall_prob):
    while True:
        dut.avm_waitrequest.value = 1 if random.random() < stall_prob else 0
        await RisingEdge(dut.clk)


async def avm_monitor(dut, writes):
    #Record every accepted transfer (write && !waitrequest), sampled
    #just before the rising edge like the rest of the suite
    while True:
        await ReadOnly()
        if dut.avm_write.value and not dut.avm_waitrequest.value:
            writes.append((int(dut.avm_address.value),
                           int(dut.avm_writedata.value)))
        await RisingEdge(dut.clk)


async def pulse_tick(dut):
    dut.frame_tick.value = 1
    await RisingEdge(dut.clk)
    dut.frame_tick.value = 0


async def wait_writes(dut, writes, count, timeout=20000):
    for _ in range(timeout):
        if len(writes) >= count:
            return
        await RisingEdge(dut.clk)
    raise AssertionError(f"timeout: {len(writes)}/{count} writes seen")


@cocotb.test()
async def tile_writer_golden_pixels(dut):
    """16 frame ticks paint all 16 tiles round-robin, exact pixel match"""
    random.seed(0)
    writes = []

    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    await reset_dut(dut)
    cocotb.start_soon(waitrequest_driver(dut, stall_prob=0.3))
    cocotb.start_soon(avm_monitor(dut, writes))

    dut.c_flat.value = pack_c_flat(TEST_MATRIX)
    dut.enable.value = 1

    pixels_per_tile = TILE_SIZE * TILE_SIZE
    expected = []
    for i in range(ARRAY_SIZE):
        for j in range(ARRAY_SIZE):
            expected += golden_tile(i, j, TEST_MATRIX[i][j])

    for tile in range(ARRAY_SIZE * ARRAY_SIZE):
        await pulse_tick(dut)
        await wait_writes(dut, writes, (tile + 1) * pixels_per_tile)

    #Idle a while: exactly one tile per tick, nothing extra
    for _ in range(100):
        await RisingEdge(dut.clk)
    assert len(writes) == len(expected), (
        f"expected {len(expected)} writes, saw {len(writes)}"
    )

    for n, (got, want) in enumerate(zip(writes, expected)):
        assert got == want, (
            f"write {n}: got addr=0x{got[0]:08x} data=0x{got[1]:04x}, "
            f"want addr=0x{want[0]:08x} data=0x{want[1]:04x}"
        )
    dut._log.info(f"all {len(writes)} pixel writes match the golden list")


@cocotb.test()
async def tile_writer_disabled_is_silent(dut):
    """enable=0 (demo mode off): ticks must not produce any writes"""
    random.seed(1)
    writes = []

    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    await reset_dut(dut)
    cocotb.start_soon(avm_monitor(dut, writes))

    dut.c_flat.value = pack_c_flat(TEST_MATRIX)
    dut.enable.value = 0

    for _ in range(4):
        await pulse_tick(dut)
        for _ in range(50):
            await RisingEdge(dut.clk)

    assert not writes, f"tile writer wrote {len(writes)} pixels while disabled"


@cocotb.test()
async def tile_writer_latches_tick(dut):
    """A tick arriving mid-tile is latched and releases the next tile"""
    random.seed(2)
    writes = []

    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    await reset_dut(dut)
    cocotb.start_soon(avm_monitor(dut, writes))

    dut.c_flat.value = pack_c_flat(TEST_MATRIX)
    dut.enable.value = 1

    pixels_per_tile = TILE_SIZE * TILE_SIZE
    await pulse_tick(dut)
    #Second tick lands while tile (0,0) is still being written
    for _ in range(10):
        await RisingEdge(dut.clk)
    await pulse_tick(dut)

    await wait_writes(dut, writes, 2 * pixels_per_tile)
    for _ in range(100):
        await RisingEdge(dut.clk)
    assert len(writes) == 2 * pixels_per_tile, (
        f"expected exactly 2 tiles ({2 * pixels_per_tile} writes), "
        f"saw {len(writes)}"
    )
