//Avalon-MM master that paints the systolic array accumulators as colored
//tiles in the VGA pixel buffer (Option B of docs/vga_demo_plan.md)
//Each frame_tick releases one tile, round-robin (tie high to free-run);
//the accumulator is snapshotted at tile start so a tile never tears
//Byte address = BASE_ADDR + (y << Y_SHIFT) + (x << 1), matching the UP
//pixel buffer DMA at 320x240 RGB565
//c_flat packs c_out[i][j] at bits [(i*ARRAY_SIZE+j+1)*ACC_WIDTH-1 -: ACC_WIDTH]
module vga_tile_writer
#(
    parameter ARRAY_SIZE  = 4,
    parameter ACC_WIDTH   = 32,
    parameter TILE_SIZE   = 16,        //pixels per tile side
    parameter TILE_GAP    = 2,         //blank pixels between tiles
    parameter ORIGIN_X    = 8,         //top-left of the grid, in pixels
    parameter ORIGIN_Y    = 8,
    parameter Y_SHIFT     = 10,        //line stride = 2^Y_SHIFT bytes
    parameter BASE_ADDR   = 32'h0,     //pixel buffer base on the avm bus
    parameter COLOR_SHIFT = 11         //|acc| >> this -> 5-bit channel ramp
)(
    input logic clk,
    input logic rst_n,

    input logic enable,     //demo_mode; when 0 the master stays idle
    input logic frame_tick, //1-cycle pulse releases the next tile
    input logic [ARRAY_SIZE*ARRAY_SIZE*ACC_WIDTH-1:0] c_flat,

    //Avalon-MM master, 16-bit data, byte addresses
    output logic [31:0] avm_address,
    output logic avm_write,
    output logic [15:0] avm_writedata,
    input logic avm_waitrequest
);

    localparam TILE_PITCH = TILE_SIZE + TILE_GAP;
    localparam IDX_WIDTH  = $clog2(ARRAY_SIZE);
    localparam PX_WIDTH   = $clog2(TILE_SIZE);

    //Mirrored by the cocotb golden model; keep in sync
    function automatic logic [15:0] colormap(input logic signed [ACC_WIDTH-1:0] acc);
        logic [ACC_WIDTH-1:0] mag;
        logic [4:0] mag5;
        mag  = acc[ACC_WIDTH-1] ? ACC_WIDTH'(-acc) : ACC_WIDTH'(acc);
        mag  = mag >> COLOR_SHIFT;
        mag5 = (mag > 31) ? 5'd31 : mag[4:0];
        if(acc == 0)
            colormap = 16'h0000;
        else if(acc[ACC_WIDTH-1])
            colormap = {11'b0, mag5};        //negative: blue ramp
        else
            colormap = {mag5, 11'b0};        //positive: red ramp
    endfunction

    typedef enum logic {
        IDLE,
        WRITE
    } state_t;
    state_t state;

    logic [IDX_WIDTH-1:0] tile_i, tile_j; //which accumulator
    logic [PX_WIDTH-1:0] px, py;          //pixel within the tile
    logic [15:0] tile_color;              //snapshot, colormapped at tile start
    logic tick_pending;                   //frame_tick latched while a tile is in flight

    logic signed [ACC_WIDTH-1:0] acc_now;
    assign acc_now = c_flat[(32'(tile_i)*ARRAY_SIZE + 32'(tile_j) + 1)*ACC_WIDTH - 1 -: ACC_WIDTH];

    logic [31:0] x_pos, y_pos;
    assign x_pos = ORIGIN_X + 32'(tile_j)*TILE_PITCH + 32'(px);
    assign y_pos = ORIGIN_Y + 32'(tile_i)*TILE_PITCH + 32'(py);

    assign avm_address   = BASE_ADDR + (y_pos << Y_SHIFT) + (x_pos << 1);
    assign avm_write     = (state == WRITE);
    assign avm_writedata = tile_color;

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state        <= IDLE;
            tile_i       <= 0;
            tile_j       <= 0;
            px           <= 0;
            py           <= 0;
            tile_color   <= 0;
            tick_pending <= 1'b0;
        end else begin
            if(frame_tick)
                tick_pending <= 1'b1;

            case(state)
                IDLE: begin
                    //enable sampled only between tiles; an in-flight
                    //Avalon write is never abandoned
                    if(enable && (frame_tick || tick_pending)) begin
                        state        <= WRITE;
                        tick_pending <= 1'b0;
                        tile_color   <= colormap(acc_now);
                        px           <= 0;
                        py           <= 0;
                    end
                end
                WRITE: begin
                    if(!avm_waitrequest) begin
                        if(32'(px) == TILE_SIZE-1) begin
                            px <= 0;
                            if(32'(py) == TILE_SIZE-1) begin
                                py    <= 0;
                                state <= IDLE;
                                if(32'(tile_j) == ARRAY_SIZE-1) begin
                                    tile_j <= 0;
                                    tile_i <= (32'(tile_i) == ARRAY_SIZE-1) ? '0 : tile_i + 1'b1;
                                end else begin
                                    tile_j <= tile_j + 1'b1;
                                end
                            end else begin
                                py <= py + 1'b1;
                            end
                        end else begin
                            px <= px + 1'b1;
                        end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
