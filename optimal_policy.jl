using Memoize
using StatsFuns: softmax
using StatsBase

struct BackwardsInduction
    m::MetaMDP
    cache::Dict{UInt64, Float64}
end
BackwardsInduction(m::MetaMDP) = BackwardsInduction(m, Dict{UInt64, Float64}())

function V(solution::BackwardsInduction, b::Belief)
    (;cache, m) = solution
    key = hash(b)
    haskey(cache, key) && return cache[key]
    return cache[key] = maximum(Q(solution, b, c) for c in computations(m, b))
end

function Q(solution::BackwardsInduction, b::Belief, c::Int)
    m = solution.m
    c == ⊥ && return term_reward(m, b)
    sum(p * V(solution, b′) for (p, b′) in transition(m, b, c)) - cost(m, b, c)
end

struct OptimalPolicy <: Policy
    m::MetaMDP
    β::Float64
    solution::BackwardsInduction
end

function sample_softmax(f::Function, x)
    sample(x, Weights(softmax(f.(x))))
end

function (policy::OptimalPolicy)(b::Belief)
    sample_softmax(computation(b)) do c
        Q(policy.solution, b, c)
    end
end
