module skew_reg
#(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8
)(
    input logic clk,
    input logic rst_n,
    input logic enable, //FSM signal

    input logic signed  [DATA_WIDTH-1:0] flat_data_in  [0:ARRAY_SIZE-1],
    output logic signed [DATA_WIDTH-1:0] skewed_data_out [0:ARRAY_SIZE-1]
);

    genvar i;
    generate
        for(i = 0; i < ARRAY_SIZE; i = i+1) begin : g_skew

            if(i == 0) begin : g_passthrough
                assign skewed_data_out[0] = flat_data_in[0];
            end else begin : g_delay_line
                //i cycle delays (green blocks on diagram)
                logic signed [DATA_WIDTH-1:0] delay [0:i-1];

                always_ff @(posedge clk or negedge rst_n) begin
                    if(!rst_n) begin
                        for(int k = 0; k < i; k=k+1) begin
                            delay[k] <= 0;
                        end
                    end else if(enable) begin
                        delay[0] <= flat_data_in[i];
                        for(int k = 1; k < i; k=k+1) begin
                            delay[k] <= delay[k-1];
                        end
                    end
                end

                assign skewed_data_out[i] = delay[i-1];
            end
        end
    endgenerate

endmodule
