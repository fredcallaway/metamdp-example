# Example code for metalevel MDPs

## Setup

1. Make sure you have a recent version of Julia installed (tested on v1.7).
2. Spin up julia and install dependencies with. The instantiate command is only necessary the first time.
```
julia --project=.
] instantiate
```
3. Check that everything is working with `include("tests.jl")`

## Files

- metamdp.jl defines the abstract MetaMDP and MetaPolicy types. All necessary methods are documented there, as well as general purpose functions e.g. for running rollouts (simulating the policy).
- tallying_mdp.jl defines the tallying example used in [my dissertation](https://fredcallaway.com/pdfs/dissertation.pdf). [Figure 2.3](policies.pdf) is generated by plot_policy.jl
- backwards_induction.jl implements recursive backwards induction for exactly solving MetaMDPs
- bmps.jl implements the BMPS algorithm for approximately solving MetaMDPs.
- bmps_demo.jl shows how BMPS can be applied to solve the bernoulli metalevel problem defined in [Hay et al. 2012](https://arxiv.org/pdf/1408.2048.pdf). It contains the implementation of VOI features for that problem.