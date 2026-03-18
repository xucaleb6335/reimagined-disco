module control_FSM
#(
parameter ARRAY_SIZE = 4,
parameter MAT_DIM = 4
) ( 
    input logic clk,
    input logic rst_
    );

    typedef enum logic [1:0] {
        IDLE,
        COMPUTE,
        DRAIN,
        STREAM_OUT
    }  state_t;

always_comb begin
    case(current_state)
            IDLE: begin end

            COMPUTE: begin end

            DRAIN: begin end

            STREAM_OUT: begin end

    endcase
end
    
endmodule
