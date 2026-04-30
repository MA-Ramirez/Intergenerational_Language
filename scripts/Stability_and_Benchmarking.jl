using DrWatson
@quickactivate "Intergenerational_Language"

include(srcdir("Model.jl"))

#EvoDyn parameters
const BETA = 1.0
const MU = 0.01
#Game parameters
const B = 4.0
const D = 3.0

const POPULATION_SIZE = 100
#number of games per strategy per WF trimming/tournament stage
const NUMBER_GAMES_PER_GENERATION = 100

# Create a model
model = init_players(42, Model_Properties(; beta=BETA, mu=MU, b=B, d=D, 
                            population_size=POPULATION_SIZE,
                            number_games_per_generation=NUMBER_GAMES_PER_GENERATION))
player = model[1]

#----------- @code_warntype --------------------
using InteractiveUtils
@code_warntype centipede_game!(player, model, Int64[])

# -----------  @btime --------------------------
using BenchmarkTools

# Simulation engine
println("Simulation engine")
@btime centipede_game!($player, $model, Int64[])

# Create a model
model = init_players(42, Model_Properties(; beta=BETA, mu=MU, b=B, d=D, 
                            population_size=POPULATION_SIZE,
                            number_games_per_generation=NUMBER_GAMES_PER_GENERATION))
player = model[1]
# Full player step (mutation + 100 games)
println("Player step")
@btime player_step!($player, $model)

# Create a model
model = init_players(42, Model_Properties(; beta=BETA, mu=MU, b=B, d=D, 
                            population_size=POPULATION_SIZE,
                            number_games_per_generation=NUMBER_GAMES_PER_GENERATION))
player = model[1]
# Fitness step (once per generation)
println("Calculate fitness linear")
@btime calculate_fitness_linear($model)


# Create a model
model = init_players(42, Model_Properties(; beta=BETA, mu=MU, b=B, d=D, 
                            population_size=POPULATION_SIZE,
                            number_games_per_generation=NUMBER_GAMES_PER_GENERATION))
player = model[1]
# Fitness step (once per generation)
println("Local mutation")
@btime change_to_local_mutation!($player, $model)


# Create a model
model = init_players(42, Model_Properties(; beta=BETA, mu=MU, b=B, d=D, 
                            population_size=POPULATION_SIZE,
                            number_games_per_generation=NUMBER_GAMES_PER_GENERATION))
player = model[1]
# Quick allocation probe
println(@allocated centipede_game!(player, model, Int64[]), " bytes")
