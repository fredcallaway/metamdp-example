using Memoize
using StatsFuns: softmax
using StatsBase

@memoize Dict function V(m::MetaMDP, b::Belief)
    maximum(Q(m, b, c) for c in computations(m, b))
end

function Q(m::MetaMDP, b::Belief, c::Int)
    c == ⊥ && return term_reward(m, b)
    sum(p * V(m, b′) for (p, b′) in transition(m, b, c)) - cost(m, b, c)
end

struct OptimalPolicy <: Policy
    m::MetaMDP
    β::Float64
end

function sample_softmax(f::Function, x)
    sample(x, Weights(softmax(f.(x))))
end

function (policy::OptimalPolicy)(b::Belief)
    sample_softmax(computation(b)) do c
        Q(policy.m, b, c)
    end
end
