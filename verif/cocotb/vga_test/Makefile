# Cocotb makefile flow for the VGA tile writer (Option B demo)
# SIM is overridable: make SIM=questa (default) or make SIM=verilator
SIM ?= questa
TOPLEVEL_LANG ?= verilog

REPO_ROOT := $(abspath $(CURDIR)/../../..)
RTL_DIR   := $(REPO_ROOT)/rtl

VERILOG_SOURCES += $(RTL_DIR)/bus_wrapper/vga_tile_writer.sv

# cocotb 1.x names
TOPLEVEL = vga_tile_writer
MODULE   = tb_vga_tile_writer
# cocotb 2.x names
COCOTB_TOPLEVEL     = vga_tile_writer
COCOTB_TEST_MODULES = tb_vga_tile_writer

ifeq ($(SIM),verilator)
    EXTRA_ARGS += --timing
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
