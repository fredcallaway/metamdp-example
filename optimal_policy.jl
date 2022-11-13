include("metamdp.jl")

using Memoize
using StatsBase: sample, Weights
using StaticArrays: SVector 

@memoize Dict function V(mdp::MetaMDP, m)
    maximum(Q(mdp, m, c) for c in computations(mdp, m))
end

function Q(mdp::MetaMDP, m, c::Int)
    c == termination_operation(mdp) && return term_reward(mdp, m)
    sum(p * V(mdp, m′) for (p, m′) in transition(mdp, m, c)) - cost(mdp, m, c)
end

struct OptimalPolicy <: MetaPolicy
    mdp::MetaMDP
    β::Float64
end

function select_computation(policy::OptimalPolicy, m)
    (;mdp, β) = policy
    argmax(computations(mdp, m)) do c
        # doctors hate this one neat trick for sampling from a softmax
        β * Q(mdp, m, c) + rand(Gumbel())
    end
end
