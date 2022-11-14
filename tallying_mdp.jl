"""
Defines a metalevel MDP for tallying.
"""

using Distributions
using Base: @kwdef
include("metamdp.jl")

@kwdef struct TallyingMDP <: MetaMDP
    cost::Float64 = 0.001
    max_step::Int = 100
    prior::Beta = Beta(1,1)
end

# ---------- World State ---------- #
WorldState = Float64  # for clarity in type annotations; not neccesary

sample_world_state(mdp::TallyingMDP) = rand(mdp.prior)

# ---------- Mental States ---------- #

struct TallyState
    time_step::Int
    heads::Int  # total positive evidence
    tails::Int  # total negative evidence
end

initial_mental_state(mdp::TallyingMDP) = TallyState(0, 0, 0)

"Distribution of w given m"
function belief(mdp::TallyingMDP, m::TallyState)
    (;α, β) = mdp.prior
    α += m.heads; β += m.tails
    Beta(α, β)
end

# ---------- Computations ---------- #
Computation = Int  # for clarity in type annotations; not neccesary

termination_operation(mdp::TallyingMDP) = 0

"Allowable computations in each belief state. Implements max_step"
function computations(mdp::TallyingMDP, m::TallyState)
    m.time_step >= mdp.max_step ? (0,) : (0, 1)
end

# ---------- Transition Function ---------- #

"Full transtion function. Samples from T(m,c,w)"
function sample_transition(mdp::TallyingMDP, m::TallyState, c::Computation, w::WorldState)
    obs_dist = Bernoulli(w)
    o = rand(obs_dist)
    bayes_update(m, c, o)
end

"Marginal transition function. Returns the distribution T(m,c)

    The distribution is represented as a tuple of (probability, next_belief) pairs.
"
function transition(mdp::TallyingMDP, m::TallyState, c::Computation)
    obs_dist = Bernoulli(mean(belief(mdp, m)))
    map((false, true)) do o
        p = pdf(obs_dist, o)
        m′ = bayes_update(m, c, o)
        (p, m′)
    end
end

"Updates the belief based on the outcome of a computation."
function bayes_update(m::TallyState, c::Computation, o::Bool)
    # we don't actually use c, but keep it in definition for consistency
    TallyState(m.time_step+1, m.heads + o, m.tails + (1-o))
end

# ---------- Reward Function ---------- #

Action = Bool
actions(mdp::TallyingMDP) = (true, false)  # accept, reject

"Cost of computation. Prettyyyy straightforward."
function cost(mdp::TallyingMDP, m::TallyState, c::Computation)
    mdp.cost
end

"Utility function U((w, a)"
function utility(mdp::TallyingMDP, w::WorldState, a::Action)
    a ? w : 1-w
end

"Expected utility function E[U(s, a) | s ~ m]"
function expected_utility(mdp::TallyingMDP, m::TallyState, a::Action)
    Es = mean(belief(mdp, m))
    utility(mdp, Es, a)  # this only works because the utility function is linear!
end
# termination reward is defined in metamdp.jl
