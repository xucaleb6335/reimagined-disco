//Top level: Avalon-MM CSR slave + AXI-Stream sink -> FIFOs -> skew regs -> array
//Usage: stream all 2*ARRAY_SIZE data beats first, then write START via the CSR
//  Beats 0..ARRAY_SIZE-1            : matrix A, one column per beat (lane i = A[i][k])
//  Beats ARRAY_SIZE..2*ARRAY_SIZE-1 : matrix B, one row per beat    (lane j = B[k][j])
//Lane n of tdata is tdata[(n+1)*DATA_WIDTH-1 -: DATA_WIDTH]
//Demo mode (CSR words 18/19): the array steps only on the demo tick and
//the tile writer paints the accumulators through the avm_* master
//(connect avm_* to the UP VGA pixel buffer slave in Qsys)
module accelerator_top #(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 6,
    parameter FIFO_DEPTH = 16,
    parameter VGA_FRAME_DIV  = 50_000, //tile writer pacing: one tile per 1 ms at 50 MHz
    parameter VGA_TILE_SIZE  = 16,
    parameter VGA_TILE_GAP   = 2,
    parameter VGA_ORIGIN_X   = 8,
    parameter VGA_ORIGIN_Y   = 8,
    parameter VGA_BASE_ADDR  = 32'h0
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
    output logic tready,

    //Avalon-MM master into the VGA pixel buffer (idle unless demo_mode)
    output logic [31:0] avm_address,
    output logic avm_write,
    output logic [15:0] avm_writedata,
    input logic avm_waitrequest
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

    //Demo mode plumbing
    logic demo_mode;
    logic [31:0] demo_div;
    logic demo_pulse;
    logic frame_pulse;
    logic tick_en;

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

    //Full speed normally; in demo mode the datapath only steps on the tick
    assign tick_en = demo_mode ? demo_pulse : 1'b1;

    demo_tick u_demo_tick (
        .clk(clk),
        .rst_n(rst_n),
        .divider(demo_div),
        .tick(demo_pulse)
    );

    control_FSM #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .MAT_DIM(ARRAY_SIZE)
    ) u_fsm (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .a_fifo_empty(a_fifo_empty),
        .b_fifo_empty(b_fifo_empty),
        .tick_en(tick_en),
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
        .c_result(c_out),
        .demo_mode(demo_mode),
        .demo_div(demo_div)
    );

    // ==========================================
    // VGA tile writer (Option B demo)
    // ==========================================
    //Flatten c_out so the tile writer port stays a plain packed vector
    logic [ARRAY_SIZE*ARRAY_SIZE*4*DATA_WIDTH-1:0] c_flat;
    genvar fi, fj;
    generate
        for(fi = 0; fi < ARRAY_SIZE; fi=fi+1) begin : g_flat_row
            for(fj = 0; fj < ARRAY_SIZE; fj=fj+1) begin : g_flat_col
                assign c_flat[(fi*ARRAY_SIZE+fj+1)*4*DATA_WIDTH-1 -: 4*DATA_WIDTH] = c_out[fi][fj];
            end
        end
    endgenerate

    demo_tick u_frame_tick (
        .clk(clk),
        .rst_n(rst_n),
        .divider(32'(VGA_FRAME_DIV)),
        .tick(frame_pulse)
    );

    vga_tile_writer #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .ACC_WIDTH(4*DATA_WIDTH),
        .TILE_SIZE(VGA_TILE_SIZE),
        .TILE_GAP(VGA_TILE_GAP),
        .ORIGIN_X(VGA_ORIGIN_X),
        .ORIGIN_Y(VGA_ORIGIN_Y),
        .BASE_ADDR(VGA_BASE_ADDR)
    ) u_tile_writer (
        .clk(clk),
        .rst_n(rst_n),
        .enable(demo_mode),
        .frame_tick(frame_pulse),
        .c_flat(c_flat),
        .avm_address(avm_address),
        .avm_write(avm_write),
        .avm_writedata(avm_writedata),
        .avm_waitrequest(avm_waitrequest)
    );

endmodule
