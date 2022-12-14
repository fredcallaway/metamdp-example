"""
Defines a metalevel MDP for the Bernoulli metalevel control problem (Hay et al. 2012)
"""

using Distributions
using Base: @kwdef
using StaticArrays

include("metamdp.jl")

# NOTE: we use a parameterized type here because the state type
# depends on the number of items and we want the state type to
# be inferrable from the MetaMDP type
@kwdef struct BernoulliProblem{N} <: MetaMDP
    cost::Float64 = 0.001
    max_step::Int = 100
    prior::Beta{Float64} = Beta(1,1)
    alternative::Float64 = 0.5
end
BernoulliProblem(N; kws...) = BernoulliProblem{N}(;kws...)
n_arm(mdp::BernoulliProblem{N}) where N = N

# ---------- World States and Actions---------- #

sample_world_state(mdp::BernoulliProblem) = ntuple(x->rand(mdp.prior), Val(n_arm(mdp)))
actions(mdp) = 0:n_arm(mdp)  # 0 means picking the alternative
arms(mdp) = 1:n_arm(mdp)

# ---------- Mental States ---------- #

struct BernoulliState{N}
    time_step::Int
    heads::NTuple{N,Int}  # total positive evidence for each item
    tails::NTuple{N,Int}  # total negative evidence for each item
end

tuple_fill(v, N) = ntuple(x->v, Val(N))

function initial_mental_state(mdp::BernoulliProblem{N}) where N
    BernoulliState(0, tuple_fill(0, N), tuple_fill(0, N))
end

function belief(mdp::BernoulliProblem, m::BernoulliState, a)
    (;α, β) = mdp.prior
    α += m.heads[a]; β += m.tails[a]
    Beta(α, β)
end

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

termination_operation(mdp) = 0

function computations(mdp::BernoulliProblem, m::BernoulliState; non_terminal=false)
    a = non_terminal ? 1 : 0
    b = m.time_step >= mdp.max_step ? 0 : n_arm(mdp)
    a:b
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

function bayes_update(m::BernoulliState, c, o::Bool)::BernoulliState
    (;time_step, heads, tails) = m
    time_step += 1
    if o
        heads = setindex(heads, heads[c] + 1, c)
    else
        tails = setindex(tails, tails[c] + 1, c)
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
