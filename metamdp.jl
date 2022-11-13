"""
Defines a metalevel MDP for tallying.
"""

using Distributions

abstract type MetaMDP{M,W,C,A} end

# ---------- World State---------- #

function sample_state(mdp::MetaMDP{M,W})::W where {M,W}
    error("sample_state(::$(typeof(mdp))) not implemented")
end


# ---------- Mental States ---------- #

struct M
    time_step::Int
    heads::Int  # total positive evidence
    tails::Int  # total negative evidence
end

function initial_mental_state(::MetaMDP{M,W})::M where {M,W}
end

function initial_mental_state(mdp::MetaMDP{M,W}, ::W)::M where {M,W}
    initial_mental_state(mdp)
end

"Distribution of w given m"
function belief(::MetaMDP{M}, ::M) where M
end

# ---------- Computations ---------- #

"The termination operation."
function termination_operation(::MetaMDP{M,W,C})::C where {M,W,C}
    # todo
end

termination_operation(::MetaMDP{M,W,Int}) where {M,W} = 0
termination_operation(::MetaMDP{M,W,String}) where {M,W} = "TERM"
termination_operation(::MetaMDP{M,W,Symbol}) where {M,W} = :TERM

"Allowable computations in each mental state. Does NOT include termination."
function computations(::MetaMDP{M,W,C}, ::M)::Tuple{Vararg{C}} where {M,W,C}
    # todo
end

# ---------- Transition Function ---------- #

"Full transtion function. Samples from T(m,c,w)"
function sample_transition(::MetaMDP{M,W,C}, ::M, c::C, w::W)::M where {M,W,C}
    # todo
end

"Marginal transition function. Returns the distribution T(m,c)

    The distribution is represented as a tuple of (probability, next_mental_state) pairs.
"
function transition(::MetaMDP{M,W,C}, ::M, ::C) where {M,W,C}
    # todo
end

# ---------- Reward Function ---------- #
"A Tuple of all possible world actions."
function actions(::MetaMDP{M,W,C,A})::Tuple{A} where {M,W,C,A}
    # todo
end

"Cost of computation. Should be positive."
function cost(::MetaMDP{M,W,C,A}, ::M, ::C)::Float64 where {M,W,C,A}
    # todo
end

"Utility function U(w, a)"
function utility(::MetaMDP{M,W,C,A}, ::W, ::A)::Float64 where {M,W,C,A}
    # todo
end

"Expected utility function E[U(w, a) | w ~ m]"
function utility(mdp::MetaMDP{M,W,C,A}, m::M, a::A)::Float64 where {M,W,C,A}
    # this default implementation assumes mean(f, B) is defined for the belief type B
    mean(belief(mdp, m)) do w
        utility(mdp, w, a)
    end
end

"Selects actions."
function action_policy(mdp::MetaMDP{M,W,C,A}, m::M)::A where {M,W,C,A}
    argmax(actions(m)) do a  # man, I love Julia
        utility(mdp, m, a)
    end
end

"Full termination reward"
function term_reward(mdp::MetaMDP{M,W}, m::M, w::W)::Float64 where {M,W}
    a = action_policy(mdp, m)
    utility(mdp, w, a)
end

"Marginal termination reward"
function term_reward(mdp::MetaMDP, m::M)
    maximum(actions(m)) do a  # seriously, are you seeing this?
        utility(mdp, b, a)
    end
end

# ---------- Policies ---------- #

abstract type MetaPolicy end

struct RandomMetaPolicy <: MetaPolicy
    mdp::MetaMDP
    p_stop::Float64
end

function select_computation(policy::RandomMetaPolicy, m)
    ⊥ = termination_operation(policy.mdp)
    rand() < policy.p_stop && return ⊥
    cs = computations(policy.mdp, m)
    isempty(cs) ? ⊥ : rand(cs)
end

function rollout(policy::MetaPolicy; mdp=policy.mdp, s=sample_state(mdp), logger=(m, c)->nothing, max_step=1000)
    m = initial_mental_state(mdp)
    ⊥ = termination_operation(mdp)
    reward = 0.
    for _ in 1:max_step
        c = select_computation(policy, m)
        logger(m, c)
        if c == ⊥
            reward += term_reward(mdp, m, s)
            return (;reward, m, s)
        else
            m = sample_transition(mdp, m, c, s)
            reward -= cost(mdp, m, c)
        end
    end
    @warn "Hit max_step"
    return (;reward, m, s)
end

# for do block syntax
rollout(logger::Function, policy; kws...) = rollout(policy; kws..., logger=logger)
