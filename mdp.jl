"""
Defines a metalevel MDP for the Bernoulli metalevel control problem (Hay et al. 2012)
"""

using Distributions
using Base: @kwdef
using Printf


@kwdef struct MetaMDP
    cost::Float64 = 0.001
    max_step::Int = 100
    prior::Beta = Beta(1,1)
end

# ---------- World State---------- #

State = Float64
sample_state(m::MetaMDP) = rand(m.prior)

# ---------- Mental States ---------- #

struct Belief
    time_step::Int
    heads::Int  # total positive evidence
    tails::Int  # total negative evidence
end

initial_belief(m::MetaMDP) = Belief(1, 0, 0)
sample_state(m::MetaMDP, b::Belief) = rand(posterior(m, b))

"Distribution of s given b"
function posterior(m::MetaMDP, b::Belief)
    (;α, β) = m.prior
    α += b.heads; β += b.tails
    Beta(α, β)
end

# ---------- Computations ---------- #

const ⊥ = 0  # termination operation

Computation = Int

"Allowable computations in each belief state. Implements max_step"
function computations(m::MetaMDP, b::Belief)
    b.time_step >= m.max_step ? (⊥,) : (⊥, 1)
end

# ---------- Transition Function ---------- #

"Full transtion function. Samples from T(b,c,s)"
function transition(m::MetaMDP, b::Belief, c::Computation, s::State)::Belief
    obs_dist = Bernoulli(s)
    o = rand(obs_dist)
    bayes_update(b, c, o)
end

"Marginal transition function. Returns the distribution T(b,c)

    The distribution is represented as a tuple of (probability, next_belief) pairs.
"
function transition(m::MetaMDP, b::Belief, c::Computation)
    obs_dist = Bernoulli(mean(posterior(m, b)))
    map((false, true)) do o
        p = pdf(obs_dist, o)
        b′ = bayes_update(b, c, o)
        (p, b′)
    end
end

"Updates the belief based on the outcome of a computation."
function bayes_update(b::Belief, c::Computation, o::Bool)::Belief
    # we don't actually use c, but keep it in definition for consistency
    Belief(b.time_step+1, b.heads + o, b.tails + (1-o))
end

# ---------- Reward Function ---------- #

Action = Bool
actions(m) = (true, false)  # accept, reject

"Cost of computation. Prettyyyy straightforward."
function cost(m::MetaMDP, b::Belief, c::Computation)::Float64
    m.cost
end

"Utility function U((s, a)"
function utility(m::MetaMDP, s::State, a::Action)
    a ? s : 1-s
end

"Expected utility function E[U(s, a) | s ~ b]"
function utility(m::MetaMDP, b::Belief, a::Action)
    Es = mean(posterior(m, b))
    utility(m, Es, a)  # this only works because the utility function is linear!
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
            reward -= cost(m, b, c)
        end
    end
end

# for do block syntax
rollout(callback::Function, policy; kws...) = rollout(policy; kws..., callback=callback)
