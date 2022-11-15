using Test
using Random
include("metamdp.jl")
include("backwards_induction.jl")
include("tallying_mdp.jl")

# The tests will sometimes fail by chance. They don't for this seed.
Random.seed!(1)

@testset "value-policy consistency" begin
    # does the optimal policy get the expected return it thinks it will?
    for i in 1:20
        mdp = TallyingMDP(cost=.01rand(), prior=Beta(5rand(), 5rand()), max_step=30)
        policy = BackwardsInduction(mdp)

        empirical = mean(1:10000) do i
            rollout(policy).reward
        end

        analytic = V(policy, initial_mental_state(mdp))

        @test empirical ≈ analytic atol=.01
    end
end

@testset "state-belief consistency" begin
    # is the belief state accurate?
    for i in 1:5
        # we build a set of lists with the states that led to each possible step-5 belief
        mdp = TallyingMDP(max_step=5, prior=Beta(10rand(), 10rand()))
        heads = 0:mdp.max_step
        states = [Float64[] for h in heads]
        nostop_policy = RandomMetaPolicy(mdp, 0.)
        for i in 1:1000000
            roll = rollout(nostop_policy)
            push!(states[roll.m.heads + 1], roll.w)
        end

        # check that the distribution of states matches the belief
        foreach(heads, states) do h, ss
            m = TallyState(mdp.max_step, h, mdp.max_step-h)
            empirical = fit(Beta, ss)
            analytic = belief(mdp, m)
            @test analytic ≈ empirical atol = 1
        end
    end
end