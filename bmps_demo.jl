include("metamdp.jl")
include("bmps.jl")
include("bernoulli_problem.jl")
include("backwards_induction.jl")

# ---------- define VOI features ---------- #

# To apply BMPS we have to define these three VOI features

function voi1(mdp::BernoulliProblem, m, c)
    sum(transition(mdp, m, c)) do (p, m′)
        p * term_reward(mdp, m′)
    end - cost(mdp, m, c)
end

function voi_action(mdp::BernoulliProblem{N}, m, c) where N
    a_considered = c
    # the SVector thing prevents allocating memory, super unnecessary optimization
    eu = SVector{N}(expected_utility(mdp, m, a) for a in arms(mdp))
    a_best, a_second = partialsortperm(eu, 1:2; rev=true)
    competing_val = a_considered == a_best ? eu[a_second] : eu[a_best]
    competing_val = max(competing_val, mdp.alternative)
    expected_max(belief(mdp, m, a_considered), competing_val)
end

function vpi(mdp::BernoulliProblem, m)
    expected_max(belief(mdp, m), mdp.alternative)
end

# HELPERS

# expected max of a Beta and a constant (could be done analytically)
@memoize function expected_max(d::Beta, constant::Float64)
    choose_new = quadgk(constant, 1., atol=1e-8) do x
        pdf(d, x) * x
    end |> first
    choose_constant = cdf(d, constant) * constant
    choose_new + choose_constant
end

# expected max of many Betas and a constant
function expected_max(pd::Product{Continuous,<:Beta}, constant::Float64=-Inf)
    components = pd.v
    function max_cdf(x)
        (constant < x) * mapreduce(*, components) do d
            cdf(d, x)
        end
    end
    lo = 0.
    hi = 1.
    # Note: this only works for non-negative RVs (need an extra term otherwise)
    quadgk(x->1-max_cdf(x), lo, hi, atol=1e-8)[1]
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
