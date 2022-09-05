using Memoize
using StatsBase: sample, Weights
using StaticArrays: SVector 

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

function (policy::OptimalPolicy)(b::Belief)
    (;m, β) = policy
    argmax(computations(m, b)) do c
        # doctors hate this one neat trick for sampling from a softmax
        β * Q(m, b, c) + rand(Gumbel())
    end
end
