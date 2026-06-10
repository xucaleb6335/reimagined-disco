//Top level: Avalon-MM CSR slave + AXI-Stream sink -> FIFOs -> skew regs -> array
//Usage: stream all 2*ARRAY_SIZE data beats first, then write START via the CSR
//  Beats 0..ARRAY_SIZE-1            : matrix A, one column per beat (lane i = A[i][k])
//  Beats ARRAY_SIZE..2*ARRAY_SIZE-1 : matrix B, one row per beat    (lane j = B[k][j])
//Lane n of tdata is tdata[(n+1)*DATA_WIDTH-1 -: DATA_WIDTH]
module accelerator_top #(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 6,
    parameter FIFO_DEPTH = 16
) (
    input logic clk,
    input logic rst_n,

    //Avalon-MM slave for control/status registers
    input logic [ADDR_WIDTH-1:0] avs_address,
    input logic avs_write,
    input logic avs_read,
    input logic [31:0] avs_writedata,
    output logic [31:0] avs_readdata,

    //AXI-Stream sink for data ingestion
    input logic [ARRAY_SIZE*DATA_WIDTH-1:0] tdata,
    input logic tvalid,
    output logic tready
);

    // ==========================================
    // Internal Wires (The "Motherboard" traces)
    // ==========================================
    //FSM control signals
    logic start;
    logic done;
    logic enable_array;
    logic pop_fifos;
    logic clear_acc;

    //FIFOs to the skew registers
    logic a_fifo_full, b_fifo_full;
    logic a_fifo_empty, b_fifo_empty;
    logic [ARRAY_SIZE*DATA_WIDTH-1:0] a_fifo_q, b_fifo_q;
    logic signed [DATA_WIDTH-1:0] a_flat [0:ARRAY_SIZE-1];
    logic signed [DATA_WIDTH-1:0] b_flat [0:ARRAY_SIZE-1];

    //Skew registers to the array
    logic signed [DATA_WIDTH-1:0] a_skewed [0:ARRAY_SIZE-1];
    logic signed [DATA_WIDTH-1:0] b_skewed [0:ARRAY_SIZE-1];

    //Array results to the CSR readout
    logic signed [4*DATA_WIDTH-1:0] c_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // ==========================================
    // AXI-Stream demux: first ARRAY_SIZE beats -> A FIFO, next -> B FIFO
    // ==========================================
    logic [$clog2(2*ARRAY_SIZE)-1:0] load_count;
    logic load_a;

    assign load_a = (load_count < ARRAY_SIZE);
    assign tready = load_a ? !a_fifo_full : !b_fifo_full;

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            load_count <= 0;
        end else if(tvalid && tready) begin
            load_count <= (load_count == 2*ARRAY_SIZE - 1) ? '0 : load_count + 1'b1;
        end
    end

    // ==========================================
    // Sub-Module Instantiations
    // ==========================================
    sync_fifo #(
        .DATA_WIDTH(ARRAY_SIZE*DATA_WIDTH),
        .DEPTH(FIFO_DEPTH)
    ) a_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(tvalid && tready && load_a),
        .wr_data(tdata),
        .full(a_fifo_full),
        .rd_en(pop_fifos),
        .rd_data(a_fifo_q),
        .empty(a_fifo_empty)
    );

    sync_fifo #(
        .DATA_WIDTH(ARRAY_SIZE*DATA_WIDTH),
        .DEPTH(FIFO_DEPTH)
    ) b_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(tvalid && tready && !load_a),
        .wr_data(tdata),
        .full(b_fifo_full),
        .rd_en(pop_fifos),
        .rd_data(b_fifo_q),
        .empty(b_fifo_empty)
    );

    //Unpack FIFO words into per-lane streams; feed zeros while draining
    genvar n;
    generate
        for(n = 0; n < ARRAY_SIZE; n=n+1) begin : g_lanes
            assign a_flat[n] = pop_fifos ? a_fifo_q[(n+1)*DATA_WIDTH-1 -: DATA_WIDTH] : '0;
            assign b_flat[n] = pop_fifos ? b_fifo_q[(n+1)*DATA_WIDTH-1 -: DATA_WIDTH] : '0;
        end
    endgenerate

    control_FSM #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .MAT_DIM(ARRAY_SIZE)
    ) u_fsm (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .a_fifo_empty(a_fifo_empty),
        .b_fifo_empty(b_fifo_empty),
        .enable_array(enable_array),
        .pop_fifos(pop_fifos),
        .clear_acc(clear_acc),
        .done(done)
    );

    skew_reg #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) a_skew (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable_array),
        .flat_data_in(a_flat),
        .skewed_data_out(a_skewed)
    );

    skew_reg #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) b_skew (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable_array),
        .flat_data_in(b_flat),
        .skewed_data_out(b_skewed)
    );

    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_array (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable_array),
        .clear_acc(clear_acc),
        .a_in(a_skewed),
        .b_in(b_skewed),
        .c_out(c_out)
    );

    avalon_csr #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_csr (
        .clk(clk),
        .rst_n(rst_n),
        .avs_address(avs_address),
        .avs_write(avs_write),
        .avs_read(avs_read),
        .avs_writedata(avs_writedata),
        .avs_readdata(avs_readdata),
        .start(start),
        .done(done),
        .c_result(c_out)
    );

endmodule
