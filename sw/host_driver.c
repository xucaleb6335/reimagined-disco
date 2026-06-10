//Host driver for the int8 systolic array accelerator (DE1-SoC, Cyclone V)
//The Avalon-MM CSR slave sits behind the Lightweight HPS-to-FPGA bridge
//
//Build (HPS Linux):  arm-linux-gnueabihf-gcc -O2 -Wall -o host_driver host_driver.c
//Build (bare-metal): add -DBARE_METAL; the physical address is used directly
//
//Register map (word addresses, matches rtl/bus_wrapper/avalon_csr.sv):
//  0x00 CTRL   : bit0 START, write 1 to start a run (self-clearing)
//  0x01 STATUS : bit0 DONE, sticky; cleared by the next START
//  0x02..      : RESULT window, C[i][j] row-major, int32 each
//
//NOTE: matrices A and B reach the accelerator through the AXI-Stream sink
//(beats 0..3 = columns of A, beats 4..7 = rows of B), e.g. via an mSGDMA
//in the Qsys system. Stream both matrices BEFORE writing START; the test
//matrices below must match what was streamed for validation to pass.

#include <stdint.h>
#include <stdio.h>

#ifndef BARE_METAL
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#define ARRAY_SIZE 4

#define LWH2F_BASE   0xFF200000u //Lightweight HPS-to-FPGA bridge
#define LWH2F_SPAN   0x00200000u
#define ACCEL_OFFSET 0x00000000u //CSR base assigned in Qsys

#define CTRL_REG     0
#define STATUS_REG   1
#define RESULT_BASE  2

#define CTRL_START   0x1u
#define STATUS_DONE  0x1u

#define DONE_TIMEOUT 1000000 //Poll iterations before giving up

static volatile uint32_t *accel; //Word pointer to the CSR block

#ifndef BARE_METAL
static int mem_fd = -1;
static void *bridge_map;
#endif

static int accel_map(void)
{
#ifdef BARE_METAL
    accel = (volatile uint32_t *)(LWH2F_BASE + ACCEL_OFFSET);
    return 0;
#else
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("open /dev/mem (are you root?)");
        return -1;
    }

    bridge_map = mmap(NULL, LWH2F_SPAN, PROT_READ | PROT_WRITE,
                      MAP_SHARED, mem_fd, LWH2F_BASE);
    if (bridge_map == MAP_FAILED) {
        perror("mmap lightweight bridge");
        close(mem_fd);
        return -1;
    }

    accel = (volatile uint32_t *)((uint8_t *)bridge_map + ACCEL_OFFSET);
    return 0;
#endif
}

static void accel_unmap(void)
{
#ifndef BARE_METAL
    if (bridge_map)
        munmap(bridge_map, LWH2F_SPAN);
    if (mem_fd >= 0)
        close(mem_fd);
#endif
}

static void accel_start(void)
{
    accel[CTRL_REG] = CTRL_START;
}

//Poll DONE; returns 0 on completion, -1 on timeout
static int accel_wait_done(void)
{
    for (int i = 0; i < DONE_TIMEOUT; i++) {
        if (accel[STATUS_REG] & STATUS_DONE)
            return 0;
    }
    return -1;
}

static void accel_read_results(int32_t c[ARRAY_SIZE][ARRAY_SIZE])
{
    for (int i = 0; i < ARRAY_SIZE; i++)
        for (int j = 0; j < ARRAY_SIZE; j++)
            c[i][j] = (int32_t)accel[RESULT_BASE + i * ARRAY_SIZE + j];
}

//Software golden reference: C = A x B with int32 accumulators
static void golden_matmul(const int8_t a[ARRAY_SIZE][ARRAY_SIZE],
                          const int8_t b[ARRAY_SIZE][ARRAY_SIZE],
                          int32_t c[ARRAY_SIZE][ARRAY_SIZE])
{
    for (int i = 0; i < ARRAY_SIZE; i++) {
        for (int j = 0; j < ARRAY_SIZE; j++) {
            c[i][j] = 0;
            for (int k = 0; k < ARRAY_SIZE; k++)
                c[i][j] += (int32_t)a[i][k] * (int32_t)b[k][j];
        }
    }
}

int main(void)
{
    //Test matrices; must match the data streamed through the AXI-Stream sink
    static const int8_t a[ARRAY_SIZE][ARRAY_SIZE] = {
        {  1,  -2,   3,  -4},
        {127, -128, 64, -64},
        {  5,   6,  -7,  -8},
        {  0,   1,   0,  -1}
    };
    static const int8_t b[ARRAY_SIZE][ARRAY_SIZE] = {
        { -1,   2,  -3,   4},
        {127,  127, -128, -128},
        { 10, -10,  20, -20},
        {  0,   1,   1,   0}
    };

    int32_t c_dut[ARRAY_SIZE][ARRAY_SIZE];
    int32_t c_gold[ARRAY_SIZE][ARRAY_SIZE];
    int errors = 0;

    if (accel_map() != 0)
        return 1;

    accel_start();

    if (accel_wait_done() != 0) {
        fprintf(stderr, "ERROR: DONE never asserted (did you stream A/B first?)\n");
        accel_unmap();
        return 1;
    }

    accel_read_results(c_dut);
    golden_matmul(a, b, c_gold);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        for (int j = 0; j < ARRAY_SIZE; j++) {
            if (c_dut[i][j] != c_gold[i][j]) {
                fprintf(stderr, "MISMATCH C[%d][%d]: FPGA=%ld golden=%ld\n",
                        i, j, (long)c_dut[i][j], (long)c_gold[i][j]);
                errors++;
            }
        }
    }

    if (errors == 0)
        printf("PASS: all %d accumulators match the golden model\n",
               ARRAY_SIZE * ARRAY_SIZE);
    else
        printf("FAIL: %d mismatches\n", errors);

    accel_unmap();
    return errors ? 1 : 0;
}
