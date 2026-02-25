//PE implementation for Systolic Array, for simulation
// EXAMPLE INSTANTIATION:
/******************************************************************************
pe #(.WIDTH(16)) pe_inst_16 (
    .clk(clk),
    .rst_n(rst_n),
    .in_a(data_a_16bit),
    .in_b(data_b_16bit),
    .out_a(out_data_a_16bit),
    .out_b(out_data_b_16bit),
    .acc(accumulator_64bit)  // 4*16 = 64 bits
);
*******************************************************************************/

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

