"""
Defines a metalevel MDP for the Bernoulli metalevel control problem (Hay et al. 2012)
"""

using Distributions
using Base: @kwdef
using StaticArrays

include("metamdp.jl")

# NOTE: we use a parameterized type here (the {N}) because the state type
# depends on the number of items and we want the state type to be inferrable
# from the MetaMDP type. Similarly, we will use SVector{N} to represent the
# world state and evidence. This fancy stuff isn't necessary, but it does
# make the code faster.

@kwdef struct BernoulliProblem{N} <: MetaMDP
    cost::Float64 = 0.001
    max_step::Int = 100
    prior::Beta{Float64} = Beta(1,1)
    alternative::Float64 = 0.5
end
BernoulliProblem(N; kws...) = BernoulliProblem{N}(;kws...)
n_arm(mdp::BernoulliProblem{N}) where N = N

# ---------- World States and Actions---------- #

function sample_world_state(mdp::BernoulliProblem{N}) where N
    # see NOTE above above SVector{N}
    SVector{N}(rand(mdp.prior, N))
end

actions(mdp) = 0:n_arm(mdp)  # 0 means picking the alternative
arms(mdp) = 1:n_arm(mdp)

# ---------- Mental States ---------- #

struct BernoulliState{N}
    time_step::Int
    heads::SVector{N,Int}  # total positive evidence for each item
    tails::SVector{N,Int}  # total negative evidence for each item
end


function initial_mental_state(mdp::BernoulliProblem{N}) where N
    init = fill(0, SVector{N})  # e.g. [0, 0, 0]
    BernoulliState(0, init, init)
end

"Belief about the value of one arm"
function belief(mdp::BernoulliProblem, m::BernoulliState, a)
    (;α, β) = mdp.prior
    α += m.heads[a]; β += m.tails[a]
    Beta(α, β)
end

"Belief about the value of all arms (the joint distribution)"
function belief(mdp::BernoulliProblem{N}, m::BernoulliState) where N
    map(SVector{N}(arms(mdp))) do a
        belief(mdp, m, a)
    end |> product_distribution
end

# This makes BackwardsInduction much faster
function symmetry_breaking_hash(m::BernoulliState)
    # double hashing is necessary b/c the sum of simple
    # hashes can ignore important order information
    mapreduce(hash ∘ hash, +, zip(m.heads, m.tails))
end

# ---------- Computations ---------- #

termination_operation(mdp::BernoulliProblem) = 0

function computations(mdp::BernoulliProblem, m::BernoulliState; non_terminal=false)
    start = non_terminal ? 1 : 0  # assume terminal operation is 0
    stop = m.time_step >= mdp.max_step ? 0 : n_arm(mdp)
    start:stop
end

# ---------- Transition Function ---------- #

function sample_transition(mdp::BernoulliProblem, m::BernoulliState, c, w)
    obs_dist = Bernoulli(w[c])
    o = rand(obs_dist)
    bayes_update(m, c, o)
end

function transition(mdp::BernoulliProblem, m::BernoulliState, c)
    obs_dist = Bernoulli(mean(belief(mdp, m, c)))
    map((false, true)) do o
        p = pdf(obs_dist, o)
        m′ = bayes_update(m, c, o)
        (p, m′)
    end
end

increment(arr, idx) = setindex(arr, arr[idx] + 1, idx)

function bayes_update(m::BernoulliState, c, o::Bool)::BernoulliState
    (;time_step, heads, tails) = m
    time_step += 1
    if o
        heads = increment(heads, c)
    else
        tails = increment(tails, c)
    end
    BernoulliState(time_step, heads, tails)
end

# ---------- Reward Function ---------- #

function cost(mdp::BernoulliProblem, m::BernoulliState, c)
    mdp.cost
end

function utility(mdp::BernoulliProblem, w, a)
    a == 0 && return mdp.alternative
    w[a]
end

function expected_utility(mdp::BernoulliProblem, m::BernoulliState, a)
    a == 0 && return mdp.alternative
    mean(belief(mdp, m, a))
end
