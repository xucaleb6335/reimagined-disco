from __future__ import annotations
import os
from pathlib import Path

from cocotb_tools.runner import get_runner

# Python runner for the single-PE cocotb test (test_pe.py)
# Usage: python systolic_arr_tb.py, or via pytest
# SIM/TOPLEVEL_LANG are overridable through the environment


def test_pe_runner():
    """Simulates the pe using the cocotb python runner"""

    hdl_toplevel_lang = os.getenv("TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "questa")

    proj_path = Path(__file__).resolve().parents[3]

    sources = [proj_path / "rtl" / "pe" / "pe.sv"]

    build_args = []
    if sim == "verilator":
        build_args += ["--timing"]

    runner = get_runner(sim)

    runner.build(
        hdl_toplevel="pe",
        sources=sources,
        parameters={"WIDTH": 8},
        build_args=build_args,
        always=True,
    )

    runner.test(
        hdl_toplevel="pe",
        hdl_toplevel_lang=hdl_toplevel_lang,
        test_module="test_pe",
        test_dir=Path(__file__).resolve().parent,
    )


if __name__ == "__main__":
    test_pe_runner()
