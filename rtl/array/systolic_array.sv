//THIS IS A 4x4 SYSTOLIC ARRAY MULTIPLIER FOR SIMULATION
//References: https://hackmd.io/@ampheo/how-to-do-systolic-architecture-on-fpga-with-verilog
//MATRIX C = A×B
module systolic_array #(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8
) (
    input logic clk,
    input logic rst_n,
    input logic enable,    //FSM signal
    input logic clear_acc,

    //LEFT edge receives A stream; one element per row and staggered
    //TOP edge receives B stream; one element per col and staggered
    input logic signed [DATA_WIDTH-1:0] a_in [0:ARRAY_SIZE-1],
    input logic signed [DATA_WIDTH-1:0] b_in [0:ARRAY_SIZE-1],

    //Partial/full sums from all PEs
    output logic signed [4*DATA_WIDTH-1:0] c_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    //Wires between PEs; a_bus[i][j]/b_bus[i][j] feed PE at row i, col j
    logic signed [DATA_WIDTH-1:0] a_bus [0:ARRAY_SIZE-1][0:ARRAY_SIZE]; //A moves right
    logic signed [DATA_WIDTH-1:0] b_bus [0:ARRAY_SIZE][0:ARRAY_SIZE-1]; //B moves down

    genvar i,j;
    generate
        //Left edge inputs assignment
        for(i = 0; i < ARRAY_SIZE; i=i+1) begin : g_left_edge
            assign a_bus[i][0] = a_in[i];
        end
        //Top edge inputs assignment
        for(j = 0; j < ARRAY_SIZE; j=j+1) begin : g_top_edge
            assign b_bus[0][j] = b_in[j];
        end

        for(i = 0; i < ARRAY_SIZE; i=i+1) begin : g_row
            for(j = 0; j < ARRAY_SIZE; j=j+1) begin : g_col
                pe #(.WIDTH(DATA_WIDTH)) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(enable),
                    .clear_acc(clear_acc),
                    .in_a(a_bus[i][j]),
                    .in_b(b_bus[i][j]),
                    .out_a(a_bus[i][j+1]), //Pass A right
                    .out_b(b_bus[i+1][j]), //Pass B downwards
                    .acc(c_out[i][j])
                );
            end
        end
    endgenerate
endmodule
