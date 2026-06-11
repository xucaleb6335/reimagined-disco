//Avalon-MM slave for control/status registers and result readout
//Register map (word addresses):
//  0x00 CTRL   : bit0 START, write 1 to start a run (self-clearing)
//  0x01 STATUS : bit0 DONE, sticky; cleared by the next START
//  0x02..0x11  : RESULT window, C[i][j] row-major, 32-bit signed each
//  0x12 DEMO_CTRL : bit0 demo_mode; gates the array on the demo tick and
//                   enables the VGA tile writer (read/write)
//  0x13 DEMO_DIV  : demo tick divider in clk cycles (read/write),
//                   resets to 5_000_000 = 10 Hz at 50 MHz
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
    input logic signed [4*DATA_WIDTH-1:0] c_result [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],

    //Demo mode
    output logic demo_mode,
    output logic [31:0] demo_div
);

    localparam CTRL_ADDR     = 0;
    localparam STATUS_ADDR   = 1;
    localparam RESULT_BASE   = 2;
    localparam RESULT_WORDS  = ARRAY_SIZE * ARRAY_SIZE;
    localparam DEMO_CTRL_ADDR = RESULT_BASE + RESULT_WORDS;     //18
    localparam DEMO_DIV_ADDR  = RESULT_BASE + RESULT_WORDS + 1; //19

    localparam DEMO_DIV_RESET = 32'd5_000_000; //10 Hz at 50 MHz

    logic done_flag;
    logic start_write;

    assign start_write = avs_write && (avs_address == CTRL_ADDR) && avs_writedata[0];

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            start     <= 1'b0;
            done_flag <= 1'b0;
            demo_mode <= 1'b0;
            demo_div  <= DEMO_DIV_RESET;
        end else begin
            start <= start_write;

            if(start_write) begin
                done_flag <= 1'b0; //New run clears DONE
            end else if(done) begin
                done_flag <= 1'b1;
            end

            if(avs_write && (avs_address == DEMO_CTRL_ADDR))
                demo_mode <= avs_writedata[0];
            if(avs_write && (avs_address == DEMO_DIV_ADDR))
                demo_div <= avs_writedata;
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
            end else if(avs_address == DEMO_CTRL_ADDR) begin
                avs_readdata <= {31'b0, demo_mode};
            end else if(avs_address == DEMO_DIV_ADDR) begin
                avs_readdata <= demo_div;
            end else begin
                avs_readdata <= 0;
            end
        end
    end
endmodule
