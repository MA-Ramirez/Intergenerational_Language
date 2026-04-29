using DrWatson
@quickactivate "Intergenerational_Language"

include(srcdir("Model.jl"))

using CSV

##################################
#           ARGUMENTS            #
##################################

const TIME_STEPS = 500
const POPULATION_SIZE = 100
#Probability epsilon to end game (10%)
const PROBABILITY_TO_END_GAME = 0.1
#number of games per strategy per WF trimming/tournament stage
const NUMBER_GAMES_PER_GENERATION = 100

#Simulation seed
#Obtain arguments for cluster run and define them as global variables
input_counter = parse(Int64,ARGS[1])

##################################################
#           MAIN FUNCTION TO RUN CODE            #
##################################################

"""
Run simulation for the parameter scan
    run_simulation(counter)
`counter`: seed for the simulation. The seed can be any natural number
The parameter scan runs the simulation for the following parameters:
    `beta`: selection intensity coefficient
    `mu`: mutation rate
The simulation results are stored in CSV file
The simulation output consists on the mean z_value of the population
"""
function run_simulation(counter)
    function initialize(; seed, beta, mu, b, d)
        return init_players(seed, Model_Properties(; beta, mu, b, d,
                population_size = POPULATION_SIZE,
                probability_to_end_game = PROBABILITY_TO_END_GAME,
                number_games_per_generation = NUMBER_GAMES_PER_GENERATION))
    end

    params = Dict(:seed => counter,
                  #Linear mapping
                  :beta => [0.01,0.2,0.4,0.6,0.8,1.0],
                  #Exponential mapping
                  #:beta => [0.01,0.1,1.0,10.0],
                  :mu => 0.01,

                  :b => 4.0,
                  :d => 3.0)

    t1 = time()

    #Perform a parameter scan of a ABM simulation
    adata, _ = paramscan(params, initialize;
        agent_step! = player_step!,
        model_step! = model_WF_step!,
        n = TIME_STEPS,
        adata=[(:z_value, mean)]
        )

    running_time = time() - t1
    println("Running time : ", running_time, " seconds")

    CSV.write(datadir("data_"*string(counter)*".csv"), adata)
end

###################################
#        RUN SIMULATION RUN       #
###################################
run_simulation(input_counter)
