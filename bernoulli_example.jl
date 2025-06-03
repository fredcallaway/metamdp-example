include("bernoulli_problem.jl")
include("backwards_induction.jl")

mdp = BernoulliProblem{2}()
policy = BackwardsInduction(mdp)
@assert V(policy, initial_mental_state(mdp)) ≈ 0.6423593964607162

roll = tracked_rollout(policy)
for (m, c) in roll.history
    print(collect(zip(m.heads, m.tails)))
    println("  ", c)
end

# %% --------

roll2 = rollout(policy, w=[0., 0.])
@assert roll2.m.heads == [0, 0]
roll2.m.tails in ([5, 15], [15, 5])  # depends on which one you happen to sample first
