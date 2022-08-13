"""
Defines a metalevel MDP for the Bernoulli metalevel control problem (Hay et al. 2012)
"""

using Distributions
using Base: @kwdef
using Printf


@kwdef struct MetaMDP
    n_action::Int = 1
    cost::Float64 = 0.001
    max_step::Int = 100
    prior::Beta = Beta(1,1)
    alternative::Float64 = 0.5
end

# ---------- World States and Actions---------- #

State = Vector{Float64}
sample_state(m::MetaMDP) = rand(m.prior, m.n_action)
Action = Int
actions(m) = 0:m.n_action  # 0 means picking the alternative

# ---------- Mental States ---------- #

mutable struct Belief
    time_step::Int
    heads::Vector{Int}  # total positive evidence for each item
    tails::Vector{Int}  # total negative evidence for each item
end
initial_belief(m::MetaMDP) = Belief(1, zeros(m.n_action), zeros(m.n_action))
sample_state(m::MetaMDP, b::Belief) = rand(posterior(m, b))
Base.copy(b::Belief) = Belief(b.time_step, copy(b.heads), copy(b.tails))

"Distribution of s[a] given b"
function posterior(m::MetaMDP, b::Belief, a::Action)
    (;α, β) = m.prior
    α += b.heads[a]; β += b.tails[a]
    Beta(α, β)
end

"Distribution of s given b"
function posterior(m::MetaMDP, b::Belief)
    map(actions(m)[2:end]) do a
        posterior(m, b, a)
    end |> product_distribution
end

# ---------- Computations ---------- #

const ⊥ = 0  # termination operation

Computation = Int

"Allowable computations in each belief state. Implements max_step"
function computations(m::MetaMDP, b::Belief)
    b.time_step >= m.max_step && return ⊥:⊥  # forced to terminate
    0:m.n_action
end

# ---------- Transition Function ---------- #

"Full transtion function. Updates the belief, sampling from T(b,c,s)"
function transition!(m::MetaMDP, b::Belief, c::Computation, s::State)::Belief
    obs_dist = Bernoulli(s[c])
    o = rand(obs_dist)
    bayes_update!(b, c, o)
end

"Marginal transition function. Returns the distribution T(b,c)

    The distribution is represented as a tuple of (probability, next_belief) pairs.
"
function transition(m::MetaMDP, b::Belief, c::Computation)
    obs_dist = Bernoulli(mean(posterior(m, b, c)))
    map((false, true)) do o
        p = pdf(obs_dist, o)
        b′ = bayes_update!(copy(b), c, o)

        (p, b′)
    end
end

"Updates the belief based on the outcome of a computation."
function bayes_update!(b::Belief, c::Computation, o::Bool)::Belief
    b.time_step += 1
    if o
        b.heads[c] += 1
    else
        b.tails[c] += 1
    end
    b
end

# ---------- Reward Function ---------- #

"Cost of computation. Prettyyyy straightforward."
function cost(m::MetaMDP, b::Belief, c::Int)::Float64
    m.cost
end

"Utility function U((s, a)"
function utility(m::MetaMDP, s::State, a::Int)
    a == 0 && return m.alternative
    s[a]
end

"Expected utility function E[U(s, a) | s ~ b]"
function utility(m::MetaMDP, b::Belief, a::Int)
    a == 0 && return m.alternative
    mean(posterior(m, b, a))
end

"Selects actions"
function action_policy(m::MetaMDP, b::Belief)
    argmax(actions(m)) do a  # man, I love Julia
        utility(m, b, a)
    end
end

"Full termination reward"
function term_reward(m::MetaMDP, b::Belief, s::State)::Float64
    a = action_policy(m, b)
    utility(m, s, a)
end

"Marginal termination reward"
function term_reward(m::MetaMDP, b::Belief)
    maximum(actions(m)) do a  # seriously, are you seeing this?
        utility(m, b, a)
    end
end

abstract type Policy end

struct RandomPolicy <: Policy
    m::MetaMDP
    p_stop::Float64
end

function (policy::RandomPolicy)(b::Belief)  # defines policy(b)
    rand() < policy.p_stop && return ⊥
    non_terminal = computations(policy.m, b)[2:end]
    isempty(non_terminal) ? ⊥ : rand(non_terminal)
end

function rollout(policy::Policy, b=initial_belief(m), s=sample_state(m, b), logger=(b, c)->nothing)
    m = policy.m
    reward = 0.
    while true
        c = policy(b)
        logger(b, c)
        if c == ⊥
            reward += term_reward(m, b, s)
            return (;reward, b, s)
        else
            transition!(m, b, c, s)
            println(b)
            reward -= cost(m, b, c)
        end
    end
end

# for do block syntax
rollout(callback::Function, policy; kws...) = rollout(policy; kws..., callback=callback)

# function Base.show(io::IO, b::Belief)
#     print(io, "[ ")
#     counts = map(1:length(b.counts)) do i
#         h, t = b.counts[i]
#         i == b.focused ? @sprintf("<%02d %02d>", h, t) : @sprintf(" %02d %02d ", h, t)
#     end
#     print(io, join(counts, " "))
#     print(io, " ]")
# end