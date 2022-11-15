include("metamdp.jl")

symmetry_breaking_hash(x) = hash(x)

struct BackwardsInduction{MDP<:MetaMDP} <: MetaPolicy
    mdp::MDP
    cache::Dict{UInt64, Float64}
end
function BackwardsInduction(mdp::MetaMDP)
    policy = BackwardsInduction(mdp, Dict{UInt64, Float64}())
    V(policy, initial_mental_state(mdp))  # solve
    policy
end
function Base.show(io::IO, policy::BackwardsInduction)
    print(io, typeof(policy), "()")
end

function V(policy::BackwardsInduction, m)
    mdp = policy.mdp
    key = symmetry_breaking_hash(m)
    haskey(policy.cache, key) && return policy.cache[key]
    v = maximum(Q(policy, m, c) for c in computations(mdp, m))
    return policy.cache[key] = v
end

function Q(policy::BackwardsInduction, m, c::Int)
    mdp = policy.mdp
    c == termination_operation(mdp) && return term_reward(mdp, m)
    sum(p * V(policy, m′) for (p, m′) in transition(mdp, m, c)) - cost(mdp, m, c)
end

function select_computation(policy::BackwardsInduction, m)
    argmax(computations(policy.mdp, m)) do c
        Q(policy, m, c)
    end
end
