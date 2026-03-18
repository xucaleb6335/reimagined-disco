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

genvar i,j;
generate
    for(int i = 0; i < ARRAY_SIZE; i = i+1 ) begin 

        if(i == 0) begin
            assign skewed_data_out[0] = flat_data_in[0];
        end else begin
                //i cycle delays
            logic signed [DATA_WIDTH-1:0] delay [0:i]; //Green blocks on diagram
                //First stage always connects to input
            assign delay[0] = flat_data_in[i]; 
                
            for(int j = 0; j < i; j=j+1) begin
                always_ff @ (posedge clk or negedge rst_n) begin

                    //Async reset could be buggy here; 
                    //Potentially switch to synchrous for synthesis
                    if(!rst_n) begin
                        delay[j+1] <= 0; //Change vector to all 0:
                    end else if(enable) begin
                        delay[j+1] = delay[j];
                    end
                end
            end
            
            assign skewed_data_out[i] = delay[i];
        end
    end
endgenerate


endmodule
