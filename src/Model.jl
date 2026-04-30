using Agents
using Random: Xoshiro
using Statistics: mean
using Distributions: Normal

#Internal simulation states
const INITIAL_SCORE = 0.0
const INITIAL_COUNT = 0
const INITIAL_FITNESS = 0.0

##################################################
#                   DEFINE AGENTS                #
##################################################

"""
Define agents 
The properties of the agents are:
`z_value`: probability to continue game. (Propensity to transmit)
`scores_sum`: each player is part of n games. Sum scores of each player for all the games they took place in.
`scores_count`: counts the amount of games the player had a score registered. 
`fitness`: player's fitness according to average array of scores in a generation
"""
@agent Player NoSpaceAgent begin
    z_value::Float64
    scores_sum::Float64
    scores_count::Int
    fitness::Float64
end

"""
Define model properties
The properties of the model are:
`beta`: selection intensity - EvoDyn
`mu`: mutation rate/probability - EvoDyn
`b`: relative advantage dominant language over heritage language - GT
`d`: long-term incremental benefit - GT
`population size`: population size of the WF model dynamics (part of `init_players`) - EvoDyn
`probability_to_end_game`: probability to end centipede due to external reasons (part of `centipede_game!`)
`number_games_per_generation`: number of games per strategy per WF trimming stage (part of `player_step!`)
`game_array`: tracker. Keeps track of agent ids in a single centipede game. Starts empty, it is changed via simulation
Note: custom mutable struct is used to avoid type instability when values are not all of the same type.
"""
Base.@kwdef mutable struct Model_Properties
    beta::Float64
    mu::Float64
    b::Float64
    d::Float64
    population_size::Int
    probability_to_end_game::Float64
    number_games_per_generation::Int
    game_array::Vector{Int} = Int[] 
end

##################################################
#                    DEFINE MODEL                #
##################################################

"""
Define and initialise the model
    init_players(seed, all_properties)
`seed`: random number generator seed
`all_properties`: model-level properties 
"""
function init_players(seed, all_properties::Model_Properties)

    #Define model
    model = StandardABM(Player; properties = all_properties, rng=Xoshiro(seed))

    #Preallocate game array (allocate once ever). Capacity of game_array persists across all games and WF stages
    sizehint!(all_properties.game_array, all_properties.population_size)

    #Add agents to the model with initialization properties
    for _ in 1:all_properties.population_size
        #z value for each player is initialised randomly from (discretized) uniform distribution
        z_value=rand(model.rng,0:0.1:1)
        add_agent!(model, z_value, INITIAL_SCORE, INITIAL_COUNT, INITIAL_FITNESS)
    end
    return model
end

##################################################
#               TIME EVOLUTION ABM               #
##################################################

#----------------------------------
#------------Player step-----------
#----------------------------------

"""
Players play the centipede game
    centipede_game(current_player, model, game_array)
Each player, namely `current_player`, starts its own centipede `game_array`
* Prob. 1-z the game ends and the current player plays NoTransmit
* Prob. epsilon the game ends (avoid infinite loops/games)
* Prob. z the game continues and random player is added to the game array, where according to its z could play Transmit or not.
The function updates the `scores_sum` and `scores_count` of the players involved in the current centipede game.
    If a player was not involved in the game, their scores are not modified.
This is the main hot loop
"""
function centipede_game!(current_player, model, game_array)
    #Bind locals once so the compiler specializes the inner arithmetic
    b = model.b
    d = model.d
    probability_to_end_game = model.probability_to_end_game

    push!(game_array, current_player.id)
    n = length(game_array)
    random_num_z = rand(model.rng)

    #TO-DO: COUNT ROUNDS PER CENTIPEDE

    #game ends with prob. 1-z
    if random_num_z > current_player.z_value

        #notransmit_me_payoff = b*(d^(n-1)) --- exponential

        notransmit_me_payoff = b+(d*(n-1))
        current_player.scores_sum += notransmit_me_payoff
        current_player.scores_count += 1
        #push!(current_player.scores, notransmit_me_payoff)

        if n > 1
            #notransmit_others_payoff = (d^(n-1)) --- exponential
            notransmit_others_payoff = (d*(n-1))

            #Last one in array is the one that breaks transmission
            for i in 1:length(game_array)-1
                id_agent = game_array[i]
                model[id_agent].scores_sum += notransmit_others_payoff
                model[id_agent].scores_count += 1
                #push!(model[i].scores, notransmit_others_payoff)
            end
        end                                                                                            
    
    #game continues with prob. z
    else
        #If game stops with probability epsilon
        random_num_end = rand(model.rng)
        if random_num_end <= probability_to_end_game
            #Everyone in the game gets
            #all_end_payoff = d^n  --- exponential
            all_end_payoff = d*n
            for i in game_array
                model[i].scores_sum += all_end_payoff
                model[i].scores_count += 1
                #push!(model[i].scores, all_end_payoff)
            end
        else
        #If game doesn't come to an end, we add another player. Game continues.
            #Elements in allids(model) that are not in game array
            available_players_ids = setdiff(allids(model), game_array)

            #if all players had played, end game
            if isempty(available_players_ids) == true
                all_end_payoff = d^n
                for i in game_array
                    model[i].scores_sum += all_end_payoff
                    model[i].scores_count += 1
                    #push!(model[i].scores, all_end_payoff)
                end 
            else
                #else continue game
                random_player_id = rand(model.rng, available_players_ids)
                centipede_game!(model[random_player_id], model, game_array)
            end
        end
    end
    #If you didnt play, nothing happens to your score
end

#---------MUTATION-----------

"""
Mutation to a random local z value
    mutate_local!(player, model)
A player could change its z value to random local mutation with probability `mu`
"""
function mutate_local!(player, model)
    if rand(model.rng) < model.mu
        change_to_local_mutation!(player,model)
    end
end

"""
Change player's z value to a local mutation
    change_to_local_mutation!(player, model)
A local mutation is drawn from a normal distribution with mean equal to the player's z value and standard deviation equal to 0.1
If the local mutation is out of the [0,1] interval, another value is drawn from the distribution
"""
function change_to_local_mutation!(player, model)
    #new_z_value = rand(model.rng, Normal(player.z_value, 0.1))
    new_z_value = round(rand(model.rng, Normal(player.z_value, 0.1)),digits=1)
    if new_z_value >= 0 && new_z_value <= 1
        player.z_value = new_z_value
    else
        change_to_local_mutation!(player, model)
    end
end

"""
Mutation to a random z value
    mutate_global!(player, model)
A player could change its z value to any other random value with probability `mu`
"""
function mutate_global!(player, model)
    if rand(model.rng) < model.mu
        #player.z_value = rand(model.rng)
        player.z_value = rand(model.rng,0:0.1:1)
    end
end

#---------MODEL STEP-----------

"""
Time evolution of ABM (players)
    player_step!(player, model)
Mutate (optional) and then play centipede games
In each ABM step, all players start one centipede game. (Each player is at least in one game per generation)
The number of centipede games played is equal to the population size. WE COULD CHANGE THIS
"""
function player_step!(player, model)
    mutate_local!(player, model)
    #mutate_global!(player, model)

    #Array with information about the game. Length is proportional to round reached
    # game_array store the ids of players who participated in the centipede game
    for i in 1:model.number_games_per_generation
        empty!(model.game_array)
        centipede_game!(player, model, model.game_array)
    end
end

#----------------------------------
#------------Model step-----------
#----------------------------------

#---------FITNESS MAPPING-----------

"""
Calculate fitness for all agents
    calculate_fitness_linear(model)
Compute fitness values via linear mapping
"""
function calculate_fitness_linear(model)
    #Bind locals once so the compiler specializes the inner arithmetic
    beta = model.beta
    
    sum_all_average_payoffs = sum((agent.scores_sum/agent.scores_count) for agent in allagents(model))
    for agent in allagents(model)
        average_payoff_agent = (agent.scores_sum/agent.scores_count)

        #Payoff normalisation
        payoff_agent = average_payoff_agent/sum_all_average_payoffs

        #Set fitness for all agents
        agent.fitness = 1 - beta + beta*payoff_agent
        #Reset scores for new generation round
        agent.scores_sum = 0.0
        agent.scores_count = 0
    end
end


"""
Calculate fitness for all agents
    calculate_fitness_exponential(model)
Compute fitness values via exponential mapping
Normalisation of average payoffs is performed to avoid numerical instability (explosion of exponential function)
"""
function calculate_fitness_exponential(model)
    #Bind locals once so the compiler specializes the inner arithmetic
    beta = model.beta

    sum_all_average_payoffs = sum((agent.scores_sum/agent.scores_count) for agent in allagents(model))
    for agent in allagents(model)
        average_payoff_agent = (agent.scores_sum/agent.scores_count)
        #Payoff normalisation
        payoff_agent = average_payoff_agent/sum_all_average_payoffs
        #Set fitness for all agents
        agent.fitness = exp(beta*payoff_agent)
        #Reset scores for new generation round
        agent.scores_sum = 0.0
        agent.scores_count = 0
    end
end

#-------------MODEL STEP------------

"""
Time evolution of ABM (model)
    model_WF_step!(model)
After a round where all players played at least one game, fitness is computed.
Afterwards, the population undergoes sampling with replacement according to the Wright-Fisher model.
"""
function model_WF_step!(model)
    #for agent in allagents(model)
    #        data = DataFrame("beta"=>[model.beta],"z_value" => [agent.z_value], "mean_score"=>[mean(agent.scores)])
    #        CSV.write(datadir("payoffs_"*string(model.beta)*".csv"), data, append=true)
    #end
    calculate_fitness_linear(model)
    sample!(model, nagents(model), :fitness)
end
