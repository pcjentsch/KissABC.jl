@everywhere pri = Factored(Normal(1, 0.5), DiscreteUniform(1, 10))
@everywhere sim((n, du)) = (n * n + du) * (n + randn() * 0.01)
@everywhere cost(x) = abs(sim(x) - 5.5)
@everywhere model_abc = ApproxPosterior(pri, cost, 0.01)
@everywhere res = sample(model_abc, AIS(100), 1000, discard_initial = 10000, progress = false)
@show sim(Tuple(res)) ≈ 5.5
res = smc(pri, cost; verbose = true)
display(res)
@show res.P[2] ≈ 5
@everywhere function brownianrms((μ, σ), N, samples = 200)
    t = 0:N
    #    rand()<1/20 && sleep(0.001)
    @.(sqrt(μ * μ * t * t + σ * σ * t)) .* (0.95 + 0.1 * rand())
end

@everywhere params = (0.5, 2.0)
@everywhere tdata = brownianrms(params, 30, 10000)
@everywhere prior = Factored(Uniform(0, 1), Uniform(0, 4))
@everywhere cost(x) = sum(abs, brownianrms(x, 30) .- tdata) / length(tdata)
@everywhere modelabc = ApproxPosterior(prior, cost, 0.1)
all(smc(prior, cost, verbose=false,min_r_ess=0.55).P .≈ params)