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
using DataFrames, CSV

using Infiltrator

# Set a random seed for reproducible behaviour
using Random
rng = Random.default_rng()
Random.seed!(rng, 000666)

tspan = [0, 130.0]
N_samples = 50 # number of points to sampler
# Times where we sample points
times_samples = sort(rand(sampler(Uniform(tspan[1], tspan[2])), N_samples))

u0 = [0.0, 0.0, -1.0]
p = 0.1 .* [1.0, 0.0, 0.0]
reltol = 1e-7
abstol = 1e-7
κ = 200 # Fisher concentration parameter on observations (small = more dispersion)

# Regularization 
do_regularization_02, λ₀₂ = false, 0.0
do_regularization_11, λ₁₁ = true,  2.0
do_regularization_12, λ₁₂ = false, 0.0

niter_ADAM  = 2000
niter_LBFGS = 500

do_regularization_DiscretizeFirst = true
Δt = 5.0        # Time step used for regularization discretization

# ###################################################
# ############    Real Solution     #################
# ###################################################

# Expected angular deviation in one unit of time (degrees)
Δω₀ = 1.0   
# Angular velocity 
ω₀ = Δω₀ * π / 180.0
# Angular momentum
τ₀ = 65.0
L0 = ω₀    .* [1.0, 0.0, 0.0]
L1 = 0.5ω₀ .* [0.0, sqrt(2), sqrt(2)]

function true_rotation!(du, u, p, t)
    if t < τ₀
        L = p[1]
    else 
        L = p[2]
    end
    du .= cross(L, u)
end

prob = ODEProblem(true_rotation!, u0, tspan, [L0, L1])
true_sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times_samples)

### Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
X_true = mapslices(x -> rand(sampler(VonMisesFisher(x/norm(x), κ)), 1), X_noiseless, dims=1)

# ###################################################
# ##############    Neural ODE    ###################
# ###################################################

# Normalization of the NN. Ideally we want to do this with L2 norm .
function sigmoid_cap(x)
    min_value = - 2ω₀
    max_value = + 2ω₀
    return min_value + (max_value - min_value) / ( 1.0 + exp(-x) )
end

function relu_cap(x)
    min_value = - 2ω₀
    max_value = + 2ω₀
    return min_value + (max_value - min_value) * max(0.0, min(x, 1.0))
end

# Define neural network 
U = Lux.Chain(
    Lux.Dense(1,5,tanh), 
    Lux.Dense(5,10,tanh), 
    Lux.Dense(10,5,tanh), 
    # Lux.Dense(5,3,sigmoid_cap)
    Lux.Dense(5,3,sigmoid_cap)
)
p, st = Lux.setup(rng, U)

function ude_rotation!(du, u, p, t)
    # Angular momentum given by network prediction
    L = U([t], p, st)[1]
    du .= cross(L, u)
    nothing
end

# ###################################################
# ###############    Training   #####################
# ###################################################

prob_nn = ODEProblem(ude_rotation!, u0, tspan, p)

function predict(θ; u0=u0, T=times_samples) 
    _prob = remake(prob_nn, u0 = u0, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Tsit5(), saveat = T,
                abstol = abstol, reltol = reltol,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss(θ)
    u_ = predict(θ)
    
    # Empirical error
    l_ = mean(abs2, u_ .- X_true)
    
    if do_regularization_DiscretizeFirst
        
        l_reg = 0.0

        # collect times for discrete regularization 
        times_reg = collect(tspan[1]:Δt:tspan[2])
        n_times = size(times_reg)[1]

        for i in 1:n_times
            t0 = times_reg[i]
            L0 = U([t0], θ, st)[1]

            # Zero-order regularization
            if do_regularization_02
                l_reg += λ₀₂ * norm(L0)^2
            end

            # First-order derivative regularization
            if (do_regularization_11 | do_regularization_12) & (i < n_times)
                t1 = times_reg[i+1]
                L1 = U([t1], θ, st)[1]
                if do_regularization_11
                    l_reg += λ₁₁ * norm(L1 .- L0)
                elseif do_regularization_12
                    l_reg += λ₁₂ * norm(L1 .- L0)^2
                end
            end    
        end
        l_reg /= n_times
        l_ += l_reg
    
    # do_regularization_DiscretizeFirst = true
    else 
        throw("Method no implemeted")
        # l_reg .+= norm(jacobian(x -> U([x], p, st)[1], t)[1])
    
    end

    return l_

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

res1 = Optimization.solve(optprob, ADAM(0.001), callback = callback, maxiters = niter_ADAM)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = niter_LBFGS)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

p_trained = res2.u
times_smooth = collect(LinRange(tspan[1], tspan[2], 1000))
u_final = predict(p_trained, T=times_smooth)


#################################################
###############   Figures   #####################
#################################################

fig = Figure(resolution=(900, 500)) 
ax = CairoMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "Value")

scatter!(ax, times_samples, X_true[1,:], label="first coordinate")#, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(ax, times_samples, X_true[2,:], label="second coordinate")#, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(ax, times_samples, X_true[3,:], label="third coordinate")

# Add legend
fig[1, 2] = Legend(fig, ax)

lines!(ax, times_smooth, u_final[1,:], label="ODE first coordinate")
lines!(ax, times_smooth, u_final[2,:], label="ODE second coordinate")
lines!(ax, times_smooth, u_final[3,:], label="ODE third coordinate")

save("Figures/sphere_solution.pdf", fig)


fig = Figure(resolution=(900, 500)) 
ax = CairoMakie.Axis(fig[1, 1], xlabel = L"Time", ylabel = "Value")

Ls = reduce(hcat, (t -> U([t], p_trained, st)[1]).(times_smooth))

scatter!(ax, times_smooth, Ls[1,:], label="ODE first coordinate")
scatter!(ax, times_smooth, Ls[2,:], label="ODE second coordinate")
scatter!(ax, times_smooth, Ls[3,:], label="ODE third coordinate")

hlines!(ax, vcat(L0, L1), 
            xmin=vcat(repeat([0.0], 3), repeat([0.5], 3)), 
            xmax=vcat(repeat([0.5], 3), repeat([1.0], 3)))
vlines!(ax, [τ₀])

fig[1, 2] = Legend(fig, ax)

save("Figures/sphere_parameter.pdf", fig)

# Save table with results 

df_data = DataFrame(time = times_samples, 
                     p1 = X_true[1,:],
                     p2 = X_true[2,:],
                     p3 = X_true[3,:]
                     )

df_res = DataFrame(time = times_smooth, 
                  u1 = u_final[1,:],
                  u2 = u_final[2,:],
                  u3 = u_final[3,:],
                  L1 = Ls[1,:],
                  L2 = Ls[2,:],
                  L3 = Ls[3,:]
                  )


CSV.write("Outputs/df_data.csv", df_data)
CSV.write("Outputs/df_results.csv", df_res)