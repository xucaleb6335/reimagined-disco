//Host driver for the int8 systolic array accelerator (DE1-SoC, Cyclone V)
//The Avalon-MM CSR slave sits behind the Lightweight HPS-to-FPGA bridge;
//the VGA dashboard (docs/vga_demo_plan.md, Option A) renders into the UP
//VGA pixel buffer mapped in vga.c
//
//Build (HPS Linux):  arm-linux-gnueabihf-gcc -O2 -Wall -o host_driver host_driver.c vga.c
//Build (bare-metal): add -DBARE_METAL; physical addresses are used directly
//Build (off-screen): add -DVGA_SIM; accelerator is stubbed, the dashboard
//                    is rendered into a heap buffer and dumped to a .ppm
//
//Usage:
//  host_driver                  validate one run, draw the dashboard
//  host_driver --bench N        batched benchmark (needs the stream hook)
//  host_driver --testpattern    VGA color bars (bring-up gate A.0)
//  host_driver --novga          console output only
//  host_driver --demo 0|1       demo mode off/on (CSR DEMO_CTRL)
//  host_driver --demodiv N      demo tick divider in 50 MHz cycles
//  host_driver --start          stream+START a run without polling DONE
//                               (use with --demo 1 to watch the wavefront)
//
//Register map (word addresses, matches rtl/bus_wrapper/avalon_csr.sv):
//  0x00 CTRL      : bit0 START, write 1 to start a run (self-clearing)
//  0x01 STATUS    : bit0 DONE, sticky; cleared by the next START
//  0x02..0x11     : RESULT window, C[i][j] row-major, int32 each
//  0x12 DEMO_CTRL : bit0 demo_mode (slow-motion array + VGA tile writer)
//  0x13 DEMO_DIV  : demo tick divider, resets to 5000000 (10 Hz)
//
//NOTE: matrices A and B reach the accelerator through the AXI-Stream sink
//(beats 0..3 = columns of A, beats 4..7 = rows of B), e.g. via an mSGDMA
//in the Qsys system; see accel_stream_ab() below.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vga.h"

#ifndef BARE_METAL
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#endif

#define ARRAY_SIZE 4
#define OPS_PER_MATMUL (2 * ARRAY_SIZE * ARRAY_SIZE * ARRAY_SIZE) //MACs x2

#define LWH2F_BASE   0xFF200000u //Lightweight HPS-to-FPGA bridge
#define LWH2F_SPAN   0x00200000u
#define ACCEL_OFFSET 0x00000000u //CSR base assigned in Qsys

#define CTRL_REG     0
#define STATUS_REG   1
#define RESULT_BASE  2
#define DEMO_CTRL_REG 18
#define DEMO_DIV_REG  19

#define CTRL_START   0x1u
#define STATUS_DONE  0x1u

#define DONE_TIMEOUT 1000000 //Poll iterations before giving up

static volatile uint32_t *accel; //Word pointer to the CSR block

//==========================================================================
// Accelerator access (stubbed under VGA_SIM for off-screen rendering)
//==========================================================================
#ifdef VGA_SIM

static uint32_t sim_regs[32];

static int accel_map(void) { accel = sim_regs; return 0; }
static void accel_unmap(void) {}

#else /* !VGA_SIM */

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

#endif /* VGA_SIM */

static void accel_start(void)
{
    accel[CTRL_REG] = CTRL_START;
}

//Poll DONE; returns 0 on completion, -1 on timeout
static int accel_wait_done(void)
{
#ifdef VGA_SIM
    return 0;
#else
    for (int i = 0; i < DONE_TIMEOUT; i++) {
        if (accel[STATUS_REG] & STATUS_DONE)
            return 0;
    }
    return -1;
#endif
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

//Test matrices; must match the data streamed through the AXI-Stream sink
static const int8_t test_a[ARRAY_SIZE][ARRAY_SIZE] = {
    {  1,  -2,   3,  -4},
    {127, -128, 64, -64},
    {  5,   6,  -7,  -8},
    {  0,   1,   0,  -1}
};
static const int8_t test_b[ARRAY_SIZE][ARRAY_SIZE] = {
    { -1,   2,  -3,   4},
    {127,  127, -128, -128},
    { 10, -10,  20, -20},
    {  0,   1,   1,   0}
};

static void accel_read_results(int32_t c[ARRAY_SIZE][ARRAY_SIZE])
{
#ifdef VGA_SIM
    golden_matmul(test_a, test_b, c);
#else
    for (int i = 0; i < ARRAY_SIZE; i++)
        for (int j = 0; j < ARRAY_SIZE; j++)
            c[i][j] = (int32_t)accel[RESULT_BASE + i * ARRAY_SIZE + j];
#endif
}

//Streaming hook: A and B reach the input FIFOs over the AXI-Stream sink,
//which the HPS cannot reach through the CSR window. Wire the Qsys mSGDMA
//(or equivalent) programming in here and return 0. Until then the driver
//assumes the data was pre-streamed externally: the first START works,
//but --bench (which must re-stream every iteration) is refused.
static int accel_stream_ab(const int8_t a[ARRAY_SIZE][ARRAY_SIZE],
                           const int8_t b[ARRAY_SIZE][ARRAY_SIZE])
{
#ifdef VGA_SIM
    (void)a; (void)b;
    return 0;
#else
    (void)a; (void)b;
    return -1; //no DMA wired yet
#endif
}

#ifndef BARE_METAL
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#else
static double now_s(void) { return 0.0; }
#endif

//==========================================================================
// Dashboard rendering (Option A)
//==========================================================================
#define HEAT_X    24  //heatmap grid origin
#define HEAT_Y    56
#define HEAT_TILE 24
#define HEAT_GAP  4
#define BAR_X     160 //throughput bars
#define BAR_WMAX  144

static void fmt_ops(double ops, char *buf, size_t len)
{
    if (ops >= 1e9)
        snprintf(buf, len, "%.1fG OPS/S", ops / 1e9);
    else if (ops >= 1e6)
        snprintf(buf, len, "%.1fM OPS/S", ops / 1e6);
    else if (ops >= 1e3)
        snprintf(buf, len, "%.1fK OPS/S", ops / 1e3);
    else
        snprintf(buf, len, "%.0f OPS/S", ops);
}

static void draw_bar(int y, const char *label, double ops, double ops_max,
                     uint16_t color)
{
    char buf[32];
    int w = ops_max > 0 ? (int)(BAR_WMAX * (ops / ops_max)) : 0;
    if (w < 2 && ops > 0)
        w = 2;

    vga_text(BAR_X, y, label, VGA_WHITE, VGA_BLACK);
    vga_fill_rect(BAR_X, y + 10, BAR_WMAX, 12, VGA_GREY);
    vga_fill_rect(BAR_X, y + 10, w, 12, color);
    fmt_ops(ops, buf, sizeof(buf));
    vga_text(BAR_X, y + 26, buf, color, VGA_BLACK);
}

static void draw_dashboard(const int32_t c[ARRAY_SIZE][ARRAY_SIZE], int pass,
                           double cpu_ops, double fpga_ops,
                           double cpu_us, double fpga_us, int bench_n)
{
    char buf[48];

    vga_clear(VGA_BLACK);
    vga_text(68, 6, "SYSTOLIC ARRAY 4X4 INT8", VGA_WHITE, VGA_BLACK);

    //PASS/FAIL banner
    vga_fill_rect(8, 20, VGA_WIDTH - 16, 16, pass ? VGA_GREEN : VGA_RED);
    vga_text((VGA_WIDTH - 4 * VGA_FONT_W) / 2, 24, pass ? "PASS" : "FAIL",
             VGA_BLACK, pass ? VGA_GREEN : VGA_RED);

    //4x4 result heatmap, one tile per accumulator
    int32_t vmax = 1;
    for (int i = 0; i < ARRAY_SIZE; i++)
        for (int j = 0; j < ARRAY_SIZE; j++) {
            int32_t mag = c[i][j] < 0 ? -c[i][j] : c[i][j];
            if (mag > vmax)
                vmax = mag;
        }
    for (int i = 0; i < ARRAY_SIZE; i++)
        for (int j = 0; j < ARRAY_SIZE; j++)
            vga_fill_rect(HEAT_X + j * (HEAT_TILE + HEAT_GAP),
                          HEAT_Y + i * (HEAT_TILE + HEAT_GAP),
                          HEAT_TILE, HEAT_TILE,
                          vga_heatmap(c[i][j], vmax));
    vga_text(HEAT_X, HEAT_Y + 4 * (HEAT_TILE + HEAT_GAP) + 4,
             "C = A X B", VGA_GREY, VGA_BLACK);

    //CPU vs FPGA throughput bars
    double ops_max = cpu_ops > fpga_ops ? cpu_ops : fpga_ops;
    draw_bar(56, "CPU", cpu_ops, ops_max, VGA_YELLOW);
    draw_bar(108, "FPGA", fpga_ops, ops_max, VGA_GREEN);

    snprintf(buf, sizeof(buf), "CPU  %.2f US/RUN", cpu_us);
    vga_text(24, 204, buf, VGA_GREY, VGA_BLACK);
    snprintf(buf, sizeof(buf), "FPGA %.2f US/RUN", fpga_us);
    vga_text(24, 216, buf, VGA_GREY, VGA_BLACK);
    snprintf(buf, sizeof(buf), "BENCH N=%d", bench_n);
    vga_text(24, 228, buf, VGA_GREY, VGA_BLACK);
}

#ifdef VGA_SIM
static int dump_ppm(const uint8_t *buf, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    fprintf(f, "P6\n%d %d\n255\n", VGA_WIDTH, VGA_HEIGHT);
    for (int y = 0; y < VGA_HEIGHT; y++) {
        for (int x = 0; x < VGA_WIDTH; x++) {
            uint16_t p;
            memcpy(&p, buf + (y << VGA_Y_SHIFT) + (x << 1), 2);
            uint8_t rgb[3] = {
                (uint8_t)(((p >> 11) & 0x1F) << 3),
                (uint8_t)(((p >> 5) & 0x3F) << 2),
                (uint8_t)((p & 0x1F) << 3),
            };
            fwrite(rgb, 1, 3, f);
        }
    }
    fclose(f);
    printf("rendered to %s\n", path);
    return 0;
}
#endif

//==========================================================================
// Main flow
//==========================================================================
int main(int argc, char **argv)
{
    int bench_n = 1;
    int testpattern = 0, novga = 0, start_only = 0;
    int demo_set = 0, demo_val = 0;
    long demodiv = -1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench") && i + 1 < argc) {
            bench_n = atoi(argv[++i]);
            if (bench_n < 1)
                bench_n = 1;
        } else if (!strcmp(argv[i], "--testpattern")) {
            testpattern = 1;
        } else if (!strcmp(argv[i], "--novga")) {
            novga = 1;
        } else if (!strcmp(argv[i], "--start")) {
            start_only = 1;
        } else if (!strcmp(argv[i], "--demo") && i + 1 < argc) {
            demo_set = 1;
            demo_val = atoi(argv[++i]) ? 1 : 0;
        } else if (!strcmp(argv[i], "--demodiv") && i + 1 < argc) {
            demodiv = atol(argv[++i]);
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    uint8_t *fb_sim = NULL;
#ifdef VGA_SIM
    int have_vga = !novga;
    if (have_vga) {
        fb_sim = calloc(1, VGA_SPAN);
        vga_attach(fb_sim);
    }
#else
    int have_vga = 0;
    if (!novga) {
        have_vga = (vga_open() == 0);
        if (!have_vga)
            fprintf(stderr, "WARNING: no VGA pixel buffer, console only\n");
    }
#endif

    //Bring-up gate A.0: static test pattern, nothing else
    if (testpattern) {
        if (!have_vga)
            return 1;
        vga_color_bars();
        printf("color bars drawn; check the monitor for a stable picture\n");
#ifdef VGA_SIM
        return dump_ppm(fb_sim, "testpattern.ppm");
#else
        vga_close();
        return 0;
#endif
    }

    if (accel_map() != 0)
        return 1;

    //Demo mode controls (Option B): set CSRs, optionally kick a run, exit
    if (demo_set || demodiv >= 0) {
        if (demodiv >= 0) {
            accel[DEMO_DIV_REG] = (uint32_t)demodiv;
            printf("DEMO_DIV = %ld cycles (%.1f Hz tick)\n",
                   demodiv, demodiv > 0 ? 50e6 / (double)demodiv : 50e6);
        }
        if (demo_set) {
            accel[DEMO_CTRL_REG] = (uint32_t)demo_val;
            printf("demo_mode %s\n", demo_val ? "ON: array tick-gated, "
                   "tile writer painting" : "OFF: full speed, VGA "
                   "released to the HPS dashboard");
        }
        if (start_only) {
            if (accel_stream_ab(test_a, test_b) != 0)
                fprintf(stderr, "NOTE: no stream hook; assuming A/B were "
                                "pre-streamed\n");
            accel_start();
            printf("START issued; not polling DONE\n");
        }
        accel_unmap();
        return 0;
    }

    int32_t c_dut[ARRAY_SIZE][ARRAY_SIZE];
    int32_t c_gold[ARRAY_SIZE][ARRAY_SIZE];
    int errors = 0;

    //--- 1. CPU golden matmul, timed over enough reps to be measurable
    int cpu_reps = bench_n > 10000 ? bench_n : 10000;
    double t0 = now_s();
    for (int n = 0; n < cpu_reps; n++)
        golden_matmul(test_a, test_b, c_gold);
    double cpu_s = (now_s() - t0) / cpu_reps;

    //--- 2. FPGA matmul: stream -> START -> poll DONE -> read, timed
    int can_stream = (accel_stream_ab(test_a, test_b) == 0);
    if (!can_stream)
        fprintf(stderr, "NOTE: no stream hook; assuming A/B were "
                        "pre-streamed (see accel_stream_ab)\n");

    t0 = now_s();
    accel_start();
    if (accel_wait_done() != 0) {
        fprintf(stderr, "ERROR: DONE never asserted (did you stream A/B "
                        "first? is demo_mode off?)\n");
        accel_unmap();
        return 1;
    }
    accel_read_results(c_dut);
    double fpga_s = now_s() - t0;

    //--- batched benchmark: amortize bridge I/O over bench_n runs
    if (bench_n > 1) {
        if (!can_stream) {
            fprintf(stderr, "--bench needs the stream hook (each run "
                            "drains the FIFOs); keeping single-run "
                            "timing\n");
        } else {
            t0 = now_s();
            for (int n = 0; n < bench_n; n++) {
                accel_stream_ab(test_a, test_b);
                accel_start();
                if (accel_wait_done() != 0) {
                    fprintf(stderr, "ERROR: DONE timeout in bench run %d\n", n);
                    accel_unmap();
                    return 1;
                }
            }
            fpga_s = (now_s() - t0) / bench_n;
        }
    }

    //--- 3. validate and draw
    for (int i = 0; i < ARRAY_SIZE; i++) {
        for (int j = 0; j < ARRAY_SIZE; j++) {
            if (c_dut[i][j] != c_gold[i][j]) {
                fprintf(stderr, "MISMATCH C[%d][%d]: FPGA=%ld golden=%ld\n",
                        i, j, (long)c_dut[i][j], (long)c_gold[i][j]);
                errors++;
            }
        }
    }

    double cpu_ops = cpu_s > 0 ? OPS_PER_MATMUL / cpu_s : 0;
    double fpga_ops = fpga_s > 0 ? OPS_PER_MATMUL / fpga_s : 0;

    if (errors == 0)
        printf("PASS: all %d accumulators match the golden model\n",
               ARRAY_SIZE * ARRAY_SIZE);
    else
        printf("FAIL: %d mismatches\n", errors);
    printf("CPU : %.3f us/run  %.3g ops/s\n", cpu_s * 1e6, cpu_ops);
    printf("FPGA: %.3f us/run  %.3g ops/s  (N=%d%s)\n", fpga_s * 1e6,
           fpga_ops, bench_n, can_stream ? "" : ", bus-overhead dominated");

    if (have_vga)
        draw_dashboard(c_dut, errors == 0, cpu_ops, fpga_ops,
                       cpu_s * 1e6, fpga_s * 1e6, bench_n);

#ifdef VGA_SIM
    if (have_vga) {
        dump_ppm(fb_sim, "dashboard.ppm");
        free(fb_sim);
    }
#else
    if (have_vga)
        vga_close();
#endif

    accel_unmap();
    return errors ? 1 : 0;
}
