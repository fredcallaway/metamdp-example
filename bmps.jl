include("metamdp.jl")
include("gp_min.jl")

using QuadGK
using Memoize
using Distributions
using Distributed
using Printf
using Sobol
using Statistics

"""
    BMPSPolicy(mdp, β, w_cost, w_voi1, w_action, w_vpi)

A metalevel policy that approximates the metalevel Q function
by a linear combination of value-of-information features.
To use this policy, you must define the voi1, voi_action,
and vpi methods for your MetaMDP.
"""
struct BMPSPolicy{MDP<:MetaMDP} <: MetaPolicy
    mdp::MDP
    β::Float64
    w_cost::Float64
    w_voi1::Float64
    w_action::Float64
    w_vpi::Float64
end


"""
    voi1(mdp::MetaMDP, m, c)

The myopic value of information. The expected termination reward
after executing c.
"""
function voi1 end

"""
    voi_action(mdp::MetaMDP, m, c)

The value of perfect information about one action. The expected
termination reward if the value of the action considered by c
is learned exactly.
"""
function voi_action end

"""
    vpi(mdp::MetaMDP, m)

The value of perfect information. The expected termination reward if the
values of all actions are learned exactly.
"""
function vpi end


function select_computation(policy::BMPSPolicy, m)
    (;mdp, β, w_cost, w_voi1, w_action, w_vpi) = policy
    cs = computations(mdp, m; non_terminal=true)
    isempty(cs) && return termination_operation(mdp)

    # NOTE: we use this complicated multi-step method to avoid computing
    # VPI when possible.

    # First identify computation using lower bound (without VPI)
    v_partial, c_i = findmax(cs) do c
        -cost(mdp, m, c) +
        -w_cost +
        w_voi1 * voi1(mdp, m, c) +
        w_action * voi_action(mdp, m, c) +
        rand(Gumbel()) / β
    end
    c = cs[c_i]
    v_term = term_reward(mdp, m) + rand(Gumbel()) / β

    # Lower bound on VOC is already better than terminating
    v_partial > v_term && return c

    # # If VPI is much slower than VOI_action, you may want to uncomment this
    # # Try putting VPI weight on VOI_action (a lower bound on VPI)
    # v_partial + w_vpi * voi_action(mdp, m, c) > v_term && return c

    if w_vpi > 0
        # Try actual VPI
        v_partial + w_vpi * vpi(mdp, m) > v_term && return c
    end

    # Nope
    return termination_operation(mdp)
end

# ---------- Optimization ---------- #

"Identifies the cost parameter that makes a hard-maximizing policy never take any computations."
@memoize function max_cost(mdp::MetaMDP)
    m0 = initial_mental_state(mdp)

    function computes(w_cost)
        policy = BMPSPolicy(mdp, Inf, w_cost, 0., 0., 1.)
        select_computation(policy, m0) != termination_operation(mdp)
    end

    w_cost = 1.
    while computes(w_cost)
        w_cost *= 2
    end

    while !computes(w_cost)
        w_cost /= 2
        if w_cost < 2^-10
            @warn "Computation is too expensive! The policy will never compute."
            return w_cost
        end
    end

    step_size = w_cost / 10
    while computes(w_cost)
        w_cost += step_size
    end
    w_cost
end

"Transforms a value from the 3D unit hybercube to weights for BMPS"
function bmps_from_hypercube(mdp::MetaMDP, β::Float64, x::AbstractVector)
    # This is a trick to go from two Uniform(0,1) samples to 
    # a unifomrm sample in the 3D simplex.
    w_voi1, w_action, w_vpi = diff([0; sort(x[2:3]); 1])
    w_cost = x[1] * max_cost(mdp)
    BMPSPolicy(mdp, β, w_cost, w_voi1, w_action, w_vpi)
end

function mean_reward(policy, n_roll, max_step, parallel)
    if parallel
        rr = @distributed (+) for i in 1:n_roll
            rollout(policy; max_step, warn_max_step=false).reward
        end
        return rr / n_roll
    else
        rr = mapreduce(+, 1:n_roll) do i
            rollout(policy; max_step, warn_max_step=false).reward
        end
        return rr / n_roll
    end
end


"""
    optimize_bmps(m::MetaMDP; β=Inf, n_iter=500, n_roll=10000, max_step=1000,
                  verbose=false, parallel=true)

Identifies a near-optimal BMPSPolicy for the given MetaMDP using Bayesian optimization.
"""
function optimize_bmps(mdp::MetaMDP; β=Inf, n_iter=500, n_roll=1000, max_step=1000,
                       verbose=false, parallel=true)
    function loss(x, nr=n_roll)
        policy = bmps_from_hypercube(mdp, β, x)
        reward, secs = @timed mean_reward(policy, n_roll, max_step, parallel)
        if verbose
            θ = [policy.w_cost, policy.w_voi1, policy.w_action, policy.w_vpi]
            println(
                "θ = ", round.(θ; digits=2),
                "   reward = ", round(reward; digits=3),
                "   seconds = ", round(secs; digits=3)
            )
            flush(stdout)
        end
        -reward
    end

    opt = gp_minimize(loss, 3, noisebounds=[-4, -2], iterations=n_iter, verbose=false)

    f_mod = loss(opt.model_optimizer, 10 * n_roll)
    f_obs = loss(opt.observed_optimizer, 10 * n_roll)
    best = f_obs < f_mod ? opt.observed_optimizer : opt.model_optimizer
    bmps_from_hypercube(mdp, β, best)
end