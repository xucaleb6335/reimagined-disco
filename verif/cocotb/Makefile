# Cocotb makefile flow for the accelerator top level
# SIM is overridable: make SIM=questa (default) or make SIM=verilator
SIM ?= questa
TOPLEVEL_LANG ?= verilog

REPO_ROOT := $(abspath $(CURDIR)/../..)
RTL_DIR   := $(REPO_ROOT)/rtl

VERILOG_SOURCES += $(RTL_DIR)/pe/pe.sv
VERILOG_SOURCES += $(RTL_DIR)/array/skew_reg.sv
VERILOG_SOURCES += $(RTL_DIR)/array/systolic_array.sv
VERILOG_SOURCES += $(RTL_DIR)/array/control_FSM.sv
VERILOG_SOURCES += $(RTL_DIR)/array/demo_tick.sv
VERILOG_SOURCES += $(RTL_DIR)/bus_wrapper/sync_fifo.sv
VERILOG_SOURCES += $(RTL_DIR)/bus_wrapper/avalon_csr.sv
VERILOG_SOURCES += $(RTL_DIR)/bus_wrapper/vga_tile_writer.sv
VERILOG_SOURCES += $(RTL_DIR)/array/accelerator_top.sv

# cocotb 1.x names
TOPLEVEL = accelerator_top
MODULE   = tb_accelerator
# cocotb 2.x names
COCOTB_TOPLEVEL     = accelerator_top
COCOTB_TEST_MODULES = tb_accelerator

ifeq ($(SIM),verilator)
    EXTRA_ARGS += --timing
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
