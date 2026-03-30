module control_FSM
#(
parameter ARRAY_SIZE = 4,
parameter MAT_DIM = 4
) ( 
    input logic clk,
    input logic rst_
    
    input logic a_fifo_empty;
    input logic b_fifo_empty;
    
    output logic enable_array;
    output logic pop_fifos;
    
    //Status signal for top-level AXI wrapper
    output logic done;
    );

    typedef enum logic [1:0] {
        IDLE,
        COMPUTE,
        DRAIN,
        STREAM_OUT
    }  state_t;
    state_t current_state,next_state;
   logic [7:0] drain_counter;
   logic [7:0] compute_counter; 


    
    always_ff  @ posedge(clk) begin
        if(!rst_n) begin
            current_state <= IDLE;

        end
        
        else begin
            current_state <= next_state;
           
            //Counters change depending on state/
            if(current_state == COMPUTE && next_state == COMPUTE) begin
                compute_counter <= compute_counter + 1;
            end 

            else if (current_state == IDLE) begin
                compute_counter <= 0;
            end

            if(current_state == DRAIN && next_state == DRAIN) begin
                drain_counter <= drain_counter + 1;
            end 

            else if (current_state == IDLE) begin
                drain_counter <= 0;
            end
        end
    end

    //Next state Logic
   always_comb begin
    case(current_state)
            IDLE: begin end
                if(!a_fifo_empty && !b_fifo_empty) begin
                    next_state = COMPUTE;
                end
            COMPUTE: begin end
                if(compute_counter == MAT_DIM - 1) begin
                    next_state = DRAIN;
                end
            DRAIN: begin end
                if(drain_counter == (2 * ARRAY_SIZE - 1)) begin
                    next_state = DONE;
                end
            STREAM_OUT: begin 
                next_state = IDLE;    
            end

     endcase
    end

    //Output Logic
    always_comb begin
        pop_fifos = 1'b0;
        enable_array = 1'b0;
        done = 1'b0;
        
        case(current_state) begin
            COMPUTE: begin
                pop_fifos = 1'b1;
                enable_array = 1'b1;
            end
            DRAIN: begin
                pop_fifos = 1'b0;
                enable_array = 1'b1;
            end
            STREAM_OUT: begin
                done = 1'b1;
            end
        endcase
    end
    
endmodule
