# HPS host driver + VGA dashboard
# `make` cross-compiles for the DE1-SoC HPS; `make sim` builds a native
# off-screen renderer (accelerator stubbed) that dumps dashboard.ppm /
# testpattern.ppm for checking layout and font without the board

CROSS  ?= arm-linux-gnueabihf-
CC      = $(CROSS)gcc
HOSTCC  = gcc
CFLAGS  = -O2 -Wall -Wextra

SRCS = host_driver.c vga.c
HDRS = vga.h

host_driver: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(SRCS)

host_driver_sim: $(SRCS) $(HDRS)
	$(HOSTCC) $(CFLAGS) -DVGA_SIM -o $@ $(SRCS)

.PHONY: sim clean
sim: host_driver_sim
	./host_driver_sim
	./host_driver_sim --testpattern

clean:
	rm -f host_driver host_driver_sim dashboard.ppm testpattern.ppm
