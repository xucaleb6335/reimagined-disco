module control_FSM
#(
    parameter ARRAY_SIZE = 4,
    parameter MAT_DIM = 4
) (
    input logic clk,
    input logic rst_n,

    input logic start, //CSR START pulse from the bus wrapper
    input logic a_fifo_empty,
    input logic b_fifo_empty,

    //Datapath clock-enable; tie high for full speed. Driving it with the
    //demo tick keeps pop_fifos/enable_array aligned by construction
    input logic tick_en,

    output logic enable_array,
    output logic pop_fifos,
    output logic clear_acc,

    //Status signal for top-level bus wrapper
    output logic done
);

    typedef enum logic [1:0] {
        IDLE,
        COMPUTE,
        DRAIN,
        STREAM_OUT
    } state_t;
    state_t current_state, next_state;
    logic [7:0] drain_counter;
    logic [7:0] compute_counter;

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            current_state   <= IDLE;
            compute_counter <= 0;
            drain_counter   <= 0;
        end else begin
            current_state <= next_state;

            //Counters only advance on gated cycles
            if(current_state == COMPUTE) begin
                if(tick_en) compute_counter <= compute_counter + 1;
            end else if(current_state == IDLE) begin
                compute_counter <= 0;
            end

            if(current_state == DRAIN) begin
                if(tick_en) drain_counter <= drain_counter + 1;
            end else if(current_state == IDLE) begin
                drain_counter <= 0;
            end
        end
    end

    //Next state Logic
    always_comb begin
        next_state = current_state;
        case(current_state)
            IDLE: begin
                //Host must load both FIFOs before pulsing START
                if(start && !a_fifo_empty && !b_fifo_empty) begin
                    next_state = COMPUTE;
                end
            end
            COMPUTE: begin
                if(tick_en && compute_counter == MAT_DIM - 1) begin
                    next_state = DRAIN;
                end
            end
            DRAIN: begin
                if(tick_en && drain_counter == (2 * ARRAY_SIZE - 1)) begin
                    next_state = STREAM_OUT;
                end
            end
            STREAM_OUT: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    //Output Logic
    always_comb begin
        pop_fifos    = 1'b0;
        enable_array = 1'b0;
        clear_acc    = 1'b0;
        done         = 1'b0;

        case(current_state)
            IDLE: begin
                //Clear accumulators only when a new run is accepted,
                //so the host can still read results while idle
                clear_acc = (next_state == COMPUTE);
            end
            COMPUTE: begin
                pop_fifos    = tick_en;
                enable_array = tick_en;
            end
            DRAIN: begin
                enable_array = tick_en;
            end
            STREAM_OUT: begin
                done = 1'b1; //1 cycle pulse; latched sticky in the CSR
            end
            default: ;
        endcase
    end

endmodule
