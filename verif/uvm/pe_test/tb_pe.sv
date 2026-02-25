//Standard test bench without using UVM
`timescale 1ns/1ps
module tb_pe;
    logic clk;
    logic rst_n;
    logic [7:0]  in_a, in_b;
    logic [31:0] in_sum;
    logic [7:0]  out_a, out_b;
    logic [31:0] out_sum;

    //DUT instantiation
    pu u_pe (
        .clk(clk),
        .rst_n(rst_n),
        .in_a(in_a), 
        .in_b(in_b), 
        .in_sum(in_sum),
        .out_a(out_a), 
        .out_b(out_b), 
        .out_sum(out_sum)
    );
    
    initial clk = 0;
    forever begin
        #5
        clk = ~clk;
    end

    //Test variables 
    int i;
    logic [7:0] rand_a, ran_b;
    initial begin

        //Initalization 
        rst_n = 0; in_a = 0; in_b = 0; in_sum = 0;
        #20 rst_n = 1;

        for(int i = 0; i < 100;i++) begin 
            rand_a = $random;
            rand_b = $random;

            @(posedge clk); //Wait 1 clock cycle
                in_a <= rand_a;
                in_b <= rand_b;
                in_sum <= 0; //could change later
            

            @(posedge clk); //Wait 1 clock cycle
            expected_sum = $signed(rand_a) * $signed(rand_b);
            
            assert(out_sum == expected_sum)
                else $error("Time %0t: mismatch A=%d B=%d | DUT=%d Ref=%d", $time, rand_a, rand_b, out_sum, expected_sum);
            end

            $display ("Test complete, 100 random vectors passed");
            $finish
        end
endmodule
