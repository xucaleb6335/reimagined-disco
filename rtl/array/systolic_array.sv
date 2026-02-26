//THIS IS A 4x4 SYSTOLIC ARRAY MULTIPLIER FOR SIMULATION
//References: https://hackmd.io/@ampheo/how-to-do-systolic-architecture-on-fpga-with-verilog
//MATRIX C = A×B
module systolic_array #(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8
) (
  input logic clk,
  input logic rst_n,

  //LEFT edge receives A stream; one element per row and staggered
  //TOP edge receives B stream; one element per col and staggered
  input logic signed [DATA_WIDTH-1:0] a_in [0:ARRAY_SIZE-1],
  input logic signed [DATA_WIDTH-1:0] b_in [0:ARRAY_SIZE-1],

  //Partial/full sums from all PEs
  output logic signed [DATA_WIDTH*2-1:0] c_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

//Wires between PEs
logic signed [DATA_WIDTH-1:0] a_bus [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
logic signed [DATA_WIDTH-1:0] b_bus [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
logic signed [2*DATA_WIDTH-1:0] acc_bus [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

genvar i,j;
generate
    //Left edge inputs assignment
    for(int i = 0; i < ARRAY_SIZE; i=i+1) begin
        assign a_bus[0][i] = a_in[i];
    end
    //Top edge inputs assignment
    for(int j = 0; j < ARRAY_SIZE; j=j+1) begin
        assign b_bus[0][j] = b_in[j];
    end

    for(int i =0 ; i < ARRAY_SIZE; i=i+1) begin
        for(int j = 0; j < ARRAY_SIZE; j=j+1) begin
            logic signed [DATA_WIDTH-1:0] a_here = a_bus[j][i];
            logic signed [DATA_WIDTH-1:0] b_here = b_bus[i][j];

            pe #(.WIDTH(8)) pe_inst_8 (
                .clk(clk),
                .rst_n(rst_n),
                .in_a(a_here),
                .in_b(b_here),
                .out_a(a_bus[j+1][i]), //Pass A right
                .out_b(b_bus[j][i+1]), //Pass B downwards
                .acc(acc_bus[i][j])  
            );

            assign c_out[i][j] = acc_bus[i][j];
        end
    end
endgenerate
endmodule