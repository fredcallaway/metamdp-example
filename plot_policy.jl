include("mdp.jl")
include("optimal_policy.jl")

using StatsPlots
using DataFrames
using Measures

gr(label="")
# %% --------

go_color = "#CCCCCC"
up_color = "#8382FF"
down_color = "#FC5B68"
edge_color = go_color

function policy_graph(m; tmax=100)
    res = Any[]
    
    todo = Set{Belief}((initial_belief(m),))
    seen = Set{Belief}()

    while !isempty(todo)
        b = pop!(todo)
        b.time_step > tmax && continue
        b in seen && continue

        push!(seen, b)
        delta = b.heads - b.tails

        if Q(m, b, 1) > term_reward(m, b)
            color = go_color
            for (p, b′) in transition(m, b, 1)
                push!(todo, b′)
            end
        else

            color = delta > 0 ? up_color : down_color
        end
        push!(res, (;b.time_step, delta, color))
    end
    DataFrame(res)
end

function plot_policy(m; kws...)
    df = policy_graph(m)
    plot()
    for row in eachrow(df)
        if row.color == go_color
            for x in (-1, 1)
                plot!([row.time_step, row.time_step+1], [row.delta, row.delta+x], color=edge_color, linewidth=1)
            end
        end
    end
    @df df scatter!(:time_step, :delta, 
        color=:color, markerstrokewidth=0, 
        # markerstrokecolor=:color, color=:white, markerstrokewidth=2,
        ylim=(-5, 5), grid=false,
        xlab="Number of Cues Considered", ylab="Relative Evidence", margin = 5mm)
    plot!(;kws...)
end

plot(
    plot_policy(MetaMDP(cost=0.01, max_step=500),
        xticks=0:5:15, xlim=(-.5,15), widen=true, title="High Cost"
    ),
    plot_policy(MetaMDP(cost=0.002, max_step=500), 
        ylab="", xticks=0:20:180, xlim=(0,80), widen=true, title="Low Cost"
    ),
    size=(800,200), 
    layout=grid(1, 2, widths=[0.3, 0.7])
)
savefig("policies.pdf")