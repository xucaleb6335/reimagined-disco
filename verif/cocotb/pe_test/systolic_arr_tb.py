from __future__ import annotations
import os
import sys
from pathlib import path

import pytest

from cocotb_tools.runner import get_runner

def test_pe_runner();
    """ Simulates the pe using python runner


    """

    hdl_toplevel_lang = os.getenv("TOPLEVEL_LANG","SystemVerilog")
    sim = os.getenv("SIM","verilator");

    proj_path = Path(__file__).resolve().parent.parent

    build_args = []

    if hdl_toplevel_lang == "SystemVerilog":
        sources = [proj_path/ "rtl" / "pe" / "pe.sv"]

        """insert check here for if sim is something idk
        if sim in .....
        """

    else:
        raise ValueError(
                f"a valid value was not provided for TOPLEVEL_LANG={hdl_toplevel_lang}, (should be systermverilog)"
        )

    extra_args = []
    parameters = {
        "in_a": 8,
        "in_b": 8,
        "out_a": 8,
        "out_b": 8
    }

    sys.path.append(str(proj_path / "tests"))

    runner = get_runner(sim)

    runner_build (
        hdl_toplevel = "accelerator_toplevel",
        source = sources,
        build_args = build_args,
        parameters = parameters,
        always = True,
    )

    runner.test(
        hdl_toplevel="accelerator_toplevel",
        hdl_toplevel_lang=hdl_toplevel_lang,
        test_module="pe_tests",
        test_args = extra_args,
    )

    if __name__ == __main__:
        test_pe_runner()
