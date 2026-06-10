//Avalon-MM slave for control/status registers and result readout
//Register map (word addresses):
//  0x00 CTRL   : bit0 START, write 1 to start a run (self-clearing)
//  0x01 STATUS : bit0 DONE, sticky; cleared by the next START
//  0x02..      : RESULT window, C[i][j] row-major, 32-bit signed each
//Reads have a fixed 1 cycle latency (registered readdata)
module avalon_csr
#(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 6
)(
    input logic clk,
    input logic rst_n,

    //Avalon-MM slave
    input logic [ADDR_WIDTH-1:0] avs_address,
    input logic avs_write,
    input logic avs_read,
    input logic [31:0] avs_writedata,
    output logic [31:0] avs_readdata,

    //Core side
    output logic start, //1 cycle pulse to the FSM
    input logic done,   //1 cycle pulse from the FSM
    input logic signed [4*DATA_WIDTH-1:0] c_result [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    localparam CTRL_ADDR    = 0;
    localparam STATUS_ADDR  = 1;
    localparam RESULT_BASE  = 2;
    localparam RESULT_WORDS = ARRAY_SIZE * ARRAY_SIZE;

    logic done_flag;
    logic start_write;

    assign start_write = avs_write && (avs_address == CTRL_ADDR) && avs_writedata[0];

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            start     <= 1'b0;
            done_flag <= 1'b0;
        end else begin
            start <= start_write;

            if(start_write) begin
                done_flag <= 1'b0; //New run clears DONE
            end else if(done) begin
                done_flag <= 1'b1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            avs_readdata <= 0;
        end else if(avs_read) begin
            if(avs_address == STATUS_ADDR) begin
                avs_readdata <= {31'b0, done_flag};
            end else if(avs_address >= RESULT_BASE
                     && avs_address < RESULT_BASE + RESULT_WORDS) begin
                avs_readdata <= 32'(c_result[(avs_address - RESULT_BASE) / ARRAY_SIZE]
                                            [(avs_address - RESULT_BASE) % ARRAY_SIZE]);
            end else begin
                avs_readdata <= 0;
            end
        end
    end
endmodule
