include("metamdp.jl")
include("bmps.jl")
include("bernoulli_problem.jl")
include("backwards_induction.jl")
using Memoize
using QuadGK
# ---------- define VOI features ---------- #

# To apply BMPS we have to define these three VOI features

# expected value of terminating after executing the computation
function voi1(mdp::BernoulliProblem, m, c)
    sum(transition(mdp, m, c)) do (p, m′)
        p * term_reward(mdp, m′)
    end
end

# expected value of resolving all uncertainty about arm c
function voi_action(mdp::BernoulliProblem{N}, m, c) where N
    max_other = maximum(
        expected_utility(mdp, m, a)
        for a in arms(mdp)
        if a != c
    )
    competing_val = max(max_other, mdp.alternative)
    expected_max(belief(mdp, m, c), competing_val)
end

# expected value of resolving all uncertainty about all arms
function vpi(mdp::BernoulliProblem, m)
    expected_max(belief(mdp, m), mdp.alternative)
end

# HELPERS

function integrate(f, lo, hi; atol=1e-8)
    quadgk(f, lo, hi; atol)[1]  #[2] is error
end

# expected max of a Beta and a constant
@memoize function expected_max(d::Beta, constant::Float64)
    expected_val_better = integrate(constant, 1.) do x
        pdf(d, x) * x
    end
    expected_val_worse = cdf(d, constant) * constant
    expected_val_better + expected_val_worse
end

# expected max of many Betas and a constant
function expected_max(pd::Product{Continuous,<:Beta}, constant::Float64=-Inf)
    # very useful trick: https://en.wikipedia.org/wiki/Expected_value#Properties
    # if the distribution can be negative, you need another term for the negative part
    upper = integrate(constant, 1.) do x
        cdf_max = mapreduce(*, pd.v) do d
            cdf(d, x)
        end
        1 - cdf_max
    end
    constant + upper
end

# ---------- test performance against optimal ---------- #

mdp = BernoulliProblem(3, max_step=50, alternative=0.5)
bmps = optimize_bmps(mdp, parallel=false, verbose=true, n_iter=100)

bmps_return = mean(1:10000) do i
    rollout(bmps).reward
end

optimal = BackwardsInduction(mdp);
optimal_return = V(optimal, initial_mental_state(mdp))

@show bmps_return optimal_return
