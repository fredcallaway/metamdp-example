"""
    Defines abstract metamdp interface.

Notation:
    - m is a mental state
    - c is a computation
    - w is a world state
    - a is a world action
"""

using Distributions

"""
    MetaMDP

Abstract type for metalevel MDPs.
"""
abstract type MetaMDP end

# ---------- World State---------- #

"""
    sample_world_state(mdp::MetaMDP)

Samples a world state.
"""
function sample_world_state end

"""
    initial_mental_state(mdp::MetaMDP)
    initial_mental_state(mdp::MetaMDP, w)

The mental state when the episode begins. Can optionally depend on the world
state.
"""
function initial_mental_state end


initial_mental_state(mdp::MetaMDP) = initial_mental_state(mdp)

"""
    belief(mdp::MetaMDP, m)

Distribution of w given m.
"""
function belief end

"""
    computations(mdp::MetaMDP, m; non_terminal=false)

Allowable computations in a mental state. This includes the
termination operation unless non_terminal=true.
"""
function computations end

"""
    termination_operation(mdp::MetaMDP)

The termination operation for the given MetaMDP.
"""
function termination_operation end



"""
    sample_transition(mdp::MetaMDP, m, c, w)

Full transtion function. Samples from T(m,c,w)
"""
function sample_transition end

"""
    transition(mdp::MetaMDP, m, c)

Marginal transition function. Returns the distribution T(m,c), represented as
a tuple of (probability, next_mental_state) pairs.
"""
function transition end

"""
    actions(mdp::MetaMDP)

A Tuple of all possible world actions.
"""
function actions end

"""
    cost(mdp::MetaMDP, m, c)

Cost of computation. Should be positive.
"""
function cost end

"""
    utility(mdp::MetaMDP, w, a)

The utility of executing an action in a world state, U(w, a)
"""
function utility end

"""
    expected_utility(mdp::MetaMDP, m, a)

Expected utility function E[U(w, a) | w ~ m].
"""
function expected_utility end

"""
    action_policy(mdp::MetaMDP, m)

Selects world actions given mental states. Called when terminating computation.
"""
function action_policy(mdp::MetaMDP, m)
    argmax(actions(mdp)) do a  # man, I love Julia
        expected_utility(mdp, m, a)
    end
end

"""
    term_reward(mdp::MetaMDP, m, w)

Full termination reward, conditional on world state.
"""
function term_reward(mdp::MetaMDP, m, w)
    a = action_policy(mdp, m)
    utility(mdp, w, a)
end

"""
    term_reward(mdp::MetaMDP, m)

Marginal termination reward.
"""
function term_reward(mdp::MetaMDP, m)
    maximum(actions(mdp)) do a  # seriously, are you seeing this?
        expected_utility(mdp, m, a)
    end
end

"Abstract type for metalevel policies."
abstract type MetaPolicy end


"""
    select_computation(policy::MetaPolicy, m)

Selects a computation to perform in the given mental state.
"""
function select_computation end

"""
    RandomMetaPolicy(mdp::MetaMDP, p_stop::Float64)

A policy that executes random computations or terminates with fixed probability.
"""
struct RandomMetaPolicy{MDP<:MetaMDP} <: MetaPolicy
    mdp::MDP
    p_stop::Float64
end

function select_computation(policy::RandomMetaPolicy, m)
    ⊥ = termination_operation(policy.mdp)
    rand() < policy.p_stop && return ⊥
    nonterminal = computations(policy.mdp, m; non_terminal=true)
    isempty(nonterminal) ? ⊥ : rand(nonterminal)
end

"""
    rollout(policy::MetaPolicy; mdp=policy.mdp, s=sample_world_state(mdp), logger=false, max_step=1000)

Runs a single episode of a policy. Optionally specify:
- an mdp (required if policy has no mdp attribute),
- a state to begin from (otherwise sampled randomly)
- a maximum number of time steps (default 1000)
- a logging function logger(m,c) to call at each step

Note that the logger can also be defined as the first argument, which allows
do block syntax. For example, to save a record of mental states and computations:

    record = []  # should add a type annotation if you need it to be fast
    rollout(policy) do m, c
        push!(record, (m, c))
    end

"""
function rollout(policy::MetaPolicy; mdp=policy.mdp, w=sample_world_state(mdp),
                 max_step=1000, logger=false, warn_max_step=true)
    m = initial_mental_state(mdp)
    ⊥ = termination_operation(mdp)
    reward = 0.
    for _ in 1:max_step
        c = select_computation(policy, m)
        logger && logger(m, c)
        if c == ⊥
            reward += term_reward(mdp, m, w)
            return (;reward, m, w)
        else
            m = sample_transition(mdp, m, c, w)
            reward -= cost(mdp, m, c)
        end
    end
    warn_max_step && @warn "Hit max_step"
    return (;reward, m, w)
end

# for do block syntax
rollout(logger::Function, policy; kws...) = rollout(policy; kws..., logger=logger)
