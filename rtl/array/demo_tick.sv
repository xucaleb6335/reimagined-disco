//Free-running clock-enable divider: 1-cycle tick pulse every `divider`
//clk cycles; divider < 2 degenerates to a tick every cycle
module demo_tick
#(
    parameter DIV_WIDTH = 32
)(
    input logic clk,
    input logic rst_n,

    input logic [DIV_WIDTH-1:0] divider, //tick period in clk cycles
    output logic tick
);

    logic [DIV_WIDTH-1:0] count;

    //'>=' so a runtime divider shrink can never strand the counter
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            count <= 0;
            tick  <= 1'b0;
        end else if(count + 1'b1 >= divider) begin
            count <= 0;
            tick  <= 1'b1;
        end else begin
            count <= count + 1'b1;
            tick  <= 1'b0;
        end
    end
endmodule
