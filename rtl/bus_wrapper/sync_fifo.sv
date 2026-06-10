//Simple synchronous show-ahead FIFO; rd_data is valid whenever !empty,
//rd_en advances to the next word
module sync_fifo
#(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 16
)(
    input logic clk,
    input logic rst_n,

    input logic wr_en,
    input logic [DATA_WIDTH-1:0] wr_data,
    output logic full,

    input logic rd_en,
    output logic [DATA_WIDTH-1:0] rd_data,
    output logic empty
);

    localparam PTR_WIDTH = $clog2(DEPTH);

    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    logic [PTR_WIDTH-1:0] wr_ptr, rd_ptr;
    logic [PTR_WIDTH:0] count;

    assign full    = (count == DEPTH);
    assign empty   = (count == 0);
    assign rd_data = mem[rd_ptr];

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            count  <= 0;
        end else begin
            if(wr_en && !full) begin
                mem[wr_ptr] <= wr_data;
                wr_ptr <= (32'(wr_ptr) == DEPTH-1) ? '0 : wr_ptr + 1'b1;
            end
            if(rd_en && !empty) begin
                rd_ptr <= (32'(rd_ptr) == DEPTH-1) ? '0 : rd_ptr + 1'b1;
            end

            case({wr_en && !full, rd_en && !empty})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end
endmodule
