# -*- coding: utf-8 -*-
using Pkg
Pkg.activate("UDESphere")

using LinearAlgebra, Statistics
using OrdinaryDiffEq
using Lux, Zygote#, Enzyme
using Optim, Optimisers
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays # for Component Array
using CairoMakie
using Distributions 

using Infiltrator

# Set a random seed for reproduceable behaviour
using Random
rng = Random.default_rng()
Random.seed!(rng, 000666)

tspan = [0, 130.0]
u0 = [0.0, 0.0, -1.0]
p = 0.1 .* [1.0, 0.0, 0.0]
reltol = 1e-7
abstol = 1e-7
κ = 10000 # Fisher concentration parameter on observations (small = more dispersion)
λ = 0.0
# λ = 1.0 # This works well! 

# ###################################################
# ############    Real Solution     #################
# ###################################################

# Expected angular deviation in one unit of time (degrees)
Δω₀ = 2.0   
# Angular velocity 
ω₀ = Δω₀ * π / 180.0
# Angular momentum
τ₀ = 65.0
L0 = ω₀    .* [1.0, 0.0, 0.0]
L1 = 0.5ω₀ .* [0.0, sqrt(2), -sqrt(2)]

function true_rotation!(du, u, p, t)
    if t < τ₀
        L = p[1]
    else 
        L = p[2]
    end
    du .= cross(L, u)
end

prob = ODEProblem(true_rotation!, u0, tspan, [L0, L1])
true_sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)
times = true_sol.t

### Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
X_true = mapslices(x -> rand(sampler(VonMisesFisher(x, κ)), 1), X_noiseless, dims=1)


fig = Figure(resolution=(900, 500)) 
ax = CairoMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "Value")

scatter!(ax, times, X_true[1,:], label="first coordinate")#, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(ax, times, X_true[2,:], label="second coordinate")#, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(ax, times, X_true[3,:], label="third coordinate")

# Add legend
fig[1, 2] = Legend(fig, ax)
# Save
# save("solution.pdf", fig)

# ###################################################
# ##############    Neural ODE    ###################
# ###################################################

# Normalization of the NN. Ideally we want to do this with L2 norm .
function sigmoid_cap(x)
    min_value = - 2ω₀
    max_value = + 2ω₀
    return min_value + (max_value - min_value) / ( 1.0 + exp(-x) )
end

# Define neural network 
U = Lux.Chain(
    Lux.Dense(1,5,tanh), 
    Lux.Dense(5,5,tanh), 
    Lux.Dense(5,3,sigmoid_cap)
)
p, st = Lux.setup(rng, U)

function ude_rotation!(du, u, p, t)
    # Angular momentum given by network prediction
    L = U([t], p, st)[1]
    du .= cross(L, u)
    nothing
end

# function loss(p)
#     prob = ODEProblem(ude_rotation!, u0, tspan, p)
#     sol = solve(prob, Tsit5(), 
#                 reltol=reltol, abstol=abstol, 
#                 sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
#     sum((sol.u[end] - [0.0, 0.0, 1.0]).^2)
# end
# @show loss(p)

# ###################################################
# ###############    Training   #####################
# ###################################################

prob_nn = ODEProblem(ude_rotation!, u0, tspan, p)

function predict(θ, u0=u0, T=times) 
    _prob = remake(prob_nn, u0 = u0, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Tsit5(), saveat = T,
                abstol = abstol, reltol = reltol,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss(θ)
    u_ = predict(θ)

    # Empirical error
    l_emp = mean(abs2, u_ .- X_true)

    # Regularization
    l_reg = 0.0
    if λ != 0.0
    times_reg = collect(tspan[1]:5.0:tspan[2])
    for i in 1:(size(times_reg)[1]-1)
        # Solution using AD
    #     # this works, but it is slow!!!
        # t = times_reg[i]
        # l_reg .+= norm(jacobian(x -> U([x], p, st)[1], t)[1]) 
        # Discrete solution
        t0 = times_reg[i]
        t1 = times_reg[i+1]
        l_reg_t = norm(U([t1], θ, st)[1] .- U([t0], θ, st)[1])
        l_reg += l_reg_t
    end
    l_reg /= size(times_reg)[1]
    end # if
    
    return l_emp + λ * l_reg
end

losses = Float64[]
callback = function (p, l)
    push!(losses, l)
    if length(losses) % 10 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

res1 = Optimization.solve(optprob, ADAM(0.002), callback = callback, maxiters = 2000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = 500)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

p_trained = res2.u
u_final = predict(p_trained)

lines!(ax, times, u_final[1,:], label="ODE first coordinate")
lines!(ax, times, u_final[2,:], label="ODE second coordinate")
lines!(ax, times, u_final[3,:], label="ODE third coordinate")

save("Figures/sphere_solution.pdf", fig)


fig = Figure(resolution=(900, 500)) 
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"Time", ylabel = "Value")

Ls = reduce(hcat, (t -> U([t], p_trained, st)[1]).(times))

scatter!(ax, times, Ls[1,:], label="ODE first coordinate")
scatter!(ax, times, Ls[2,:], label="ODE second coordinate")
scatter!(ax, times, Ls[3,:], label="ODE third coordinate")

hlines!(ax, vcat(L0, L1), 
            xmin=vcat(repeat([0.0], 3), repeat([0.5], 3)), 
            xmax=vcat(repeat([0.5], 3), repeat([1.0], 3)))
vlines!(ax, [τ₀])

fig[1, 2] = Legend(fig, ax)

save("Figures/sphere_parameter.pdf", fig)
