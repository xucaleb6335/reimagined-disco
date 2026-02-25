//PE implementation for Systolic Array
//Using this for simulation for now
module pe #(parameter WIDTH = 8) (
    input logic clk,
    input logic rst_n,
    input logic signed [WIDTH-1:0] in_a,
    input logic signed [WIDTH-1:0] in_b,
    output logic signed [WIDTH-1:0] out_a,
    output logic signed [WIDTH-1:0] out_b,
    output logic signed [4*WIDTH-1:0] acc //Accumulated result
);

    always_ff @(posedge clk or posedge rst) begin
        if(!rst) begin 
            out_a <= 0;
            out_b <= 0;
            acc   <= 0;
        end else begin
            acc <= acc + (in_a*in_b);
            out_a <= in_a;
            out_b <= in_b;
        end
    end
endmodule

