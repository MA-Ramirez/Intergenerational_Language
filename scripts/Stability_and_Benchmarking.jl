using DrWatson
@quickactivate "Intergenerational_Language"

include(srcdir("Model.jl"))

# Create a model
model = init_players(42, (mu = 0.01, beta = 1.0))
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
model = init_players(42, (mu = 0.01, beta = 1.0))
player = model[1]
# Full player step (mutation + 100 games)
println("Player step")
@btime player_step!($player, $model)

# Create a model
model = init_players(42, (mu = 0.01, beta = 1.0))
player = model[1]
# Fitness step (once per generation)
println("Calculate fitness linear")
@btime calculate_fitness_linear($model)


# Create a model
model = init_players(42, (mu = 0.01, beta = 1.0))
player = model[1]
# Quick allocation probe
println(@allocated centipede_game!(player, model, Int64[]), " bytes")