//Standard test bench without using UVM
`timescale 1ns/1ps
module tb_pe;
    logic clk;
    logic rst_n;
    logic enable, clear_acc;
    logic signed [7:0]  in_a, in_b;
    logic signed [7:0]  out_a, out_b;
    logic signed [31:0] acc;

    //Reference model
    logic signed [31:0] expected_acc;

    //DUT instantiation
    pe u_pe (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .clear_acc(clear_acc),
        .in_a(in_a),
        .in_b(in_b),
        .out_a(out_a),
        .out_b(out_b),
        .acc(acc)
    );

    initial begin
        clk = 0;
        forever begin
            #5 clk = ~clk;
        end
    end

    //Test variables
    logic signed [7:0] rand_a, rand_b;
    initial begin

        //Initialization
        rst_n = 0; enable = 0; clear_acc = 0; in_a = 0; in_b = 0;
        expected_acc = 0;
        #20 rst_n = 1;
        enable = 1;

        for(int i = 0; i < 100; i++) begin
            rand_a = 8'($random);
            rand_b = 8'($random);

            in_a = rand_a;
            in_b = rand_b;
            @(posedge clk); //DUT accumulates these operands at this edge
            expected_acc = expected_acc + rand_a*rand_b;

            #1; //Let non-blocking updates settle
            assert(acc == expected_acc)
                else $error("Time %0t: mismatch A=%0d B=%0d | DUT=%0d Ref=%0d", $time, rand_a, rand_b, acc, expected_acc);
        end

        //clear_acc wipes the accumulator without a reset
        clear_acc = 1;
        @(posedge clk);
        #1;
        assert(acc == 0)
            else $error("Time %0t: clear_acc failed, acc=%0d", $time, acc);

        $display("Test complete, 100 random vectors passed");
        $finish;
    end
endmodule
