var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = KissABC","category":"page"},{"location":"#KissABC-3.0","page":"Home","title":"KissABC 3.0","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [KissABC]","category":"page"},{"location":"#KissABC.ApproxKernelizedPosterior","page":"Home","title":"KissABC.ApproxKernelizedPosterior","text":"ApproxKernelizedPosterior(\n    prior::Distribution,\n    cost::Function,\n    target_average_cost::Real\n)\n\nthis function will return a type which can be used in the sample function as an ABC density, this type works by assuming Gaussianly distributed errors 𝒩(0,ϵ), ϵ is specified in the variable target_average_cost.\n\n\n\n\n\n","category":"type"},{"location":"#KissABC.ApproxPosterior","page":"Home","title":"KissABC.ApproxPosterior","text":"ApproxPosterior(\n    prior::Distribution,\n    cost::Function,\n    max_cost::Real\n)\n\nthis function will return a type which can be used in the sample function as an ABC density, this type works by assuming uniformly distributed errors in [-ϵ,ϵ], ϵ is specified in the variable max_cost.\n\n\n\n\n\n","category":"type"},{"location":"#KissABC.CommonLogDensity","page":"Home","title":"KissABC.CommonLogDensity","text":"CommonLogDensity(nparameters, sample_init, lπ)\n\nthis function will return a type for performing classical MCMC via the sample function.\n\nnparameters: total number of parameters per sample.\n\nsample_init: function which accepts an RNG::AbstractRNG and returns a sample for lπ.\n\nlπ: function which accepts a sample, and returns a log-density float value.\n\n\n\n\n\n","category":"type"},{"location":"#KissABC.Factored","page":"Home","title":"KissABC.Factored","text":"Factored{N} <: Distribution{Multivariate, MixedSupport}\n\na Distribution type that can be used to combine multiple UnivariateDistribution's and sample from them. Example: it can be used as prior = Factored(Normal(0,1), Uniform(-1,1))\n\n\n\n\n\n","category":"type"},{"location":"#Base.length-Union{Tuple{Factored{N}}, Tuple{N}} where N","page":"Home","title":"Base.length","text":"length(p::Factored) = begin\n\nreturns the number of distributions contained in p.\n\n\n\n\n\n","category":"method"},{"location":"#Base.rand-Union{Tuple{N}, Tuple{Random.AbstractRNG, Factored{N}}} where N","page":"Home","title":"Base.rand","text":"rand(rng::AbstractRNG, factoreddist::Factored)\n\nfunction to sample one element from a Factored object\n\n\n\n\n\n","category":"method"},{"location":"#Distributions.logpdf-Union{Tuple{N}, Tuple{Factored{N}, Any}} where N","page":"Home","title":"Distributions.logpdf","text":"logpdf(d::Factored, x) = begin\n\nFunction to evaluate the logpdf of a Factored distribution object\n\n\n\n\n\n","category":"method"},{"location":"#Distributions.pdf-Union{Tuple{N}, Tuple{Factored{N}, Any}} where N","page":"Home","title":"Distributions.pdf","text":"pdf(d::Factored, x) = begin\n\nFunction to evaluate the pdf of a Factored distribution object\n\n\n\n\n\n","category":"method"},{"location":"#KissABC.cdf_g_inv-Tuple{Any, Any}","page":"Home","title":"KissABC.cdf_g_inv","text":"Inverse cdf of g-pdf, see eq. 10 of Foreman-Mackey et al. 2013.\n\n\n\n\n\n","category":"method"},{"location":"#KissABC.sample_g-Tuple{Random.AbstractRNG, Any}","page":"Home","title":"KissABC.sample_g","text":"Sample from g using inverse transform sampling.  a=2.0 is recommended.\n\n\n\n\n\n","category":"method"},{"location":"#KissABC.smc-Union{Tuple{Tprior}, Tuple{Tprior, Any}} where Tprior<:Distribution","page":"Home","title":"KissABC.smc","text":"Adaptive SMC from P. Del Moral 2012, with Affine invariant proposal mechanism, faster that AIS for ABC targets.\n\nfunction smc(\n    prior::Distribution,\n    cost::Function;\n    rng::AbstractRNG = Random.GLOBAL_RNG,\n    nparticles::Int = 100,\n    M::Int = 1,\n    alpha = 0.95,\n    mcmc_retrys::Int = 0,\n    mcmc_tol = 0.015,\n    epstol = 0.0,\n    r_epstol = (1 - alpha)^1.5 / 50,\n    min_r_ess = alpha^2,\n    max_stretch = 2.0,\n    verbose::Bool = false,\n    parallel::Bool = false,\n)\n\nprior: a Distribution object representing the parameters prior.\ncost: a function that given a prior sample returns the cost for said sample (e.g. a distance between simulated data and target data).\nrng: an AbstractRNG object which is used by SMC for inference (it can be useful to make an inference reproducible).\nnparticles: number of total particles to use for inference.\nM: number of cost evaluations per particle, increasing this can reduce the chance of rejecting good particles. \nalpha - used for adaptive tolerance, by solving ESS(n,ϵ(n)) = α ESS(n-1, ϵ(n-1)) for ϵ(n) at step n.\nmcmc_retrys - if set > 0, whenever the fraction of accepted particles drops below the tolerance mcmc_tol the MCMC step is repeated (no more than mcmc_retrys times).\nmcmc_tol - stopping condition for SMC, if the fraction of accepted particles drops below mcmc_tol the algorithm terminates.\nepstol - stopping condition for SMC, if the adaptive cost threshold drops below epstol the algorithm has converged and thus it terminates.\nmin_r_ess - whenever the fractional effective sample size drops below min_r_ess, a systematic resampling step is performed.\nmax_stretch - the proposal distribution of smc is the stretch move of Foreman-Mackey et al. 2013, the larger the parameters the wider becomes the proposal distribution.\nverbose - if set to true, enables verbosity.\nparallel - if set to true, threaded parallelism is enabled, keep in mind that the cost function must be Thread-safe in such case.\n\nExample\n\nusing KissABC\nprior=Factored(Normal(0,5), Normal(0,5))\ncost((x,y)) = 50*(x+randn()*0.01-y^2)^2+(y-1+randn()*0.01)^2\nresults = smc(prior, cost, alpha=0.5, nparticles=5000).P\n\noutput:\n\n2-element Array{Particles{Float64,5000},1}:\n 1.0 ± 0.029\n 0.999 ± 0.012\n\n\n\n\n\n","category":"method"},{"location":"#StatsBase.sample","page":"Home","title":"StatsBase.sample","text":"sample(model, AIS(N), Ns[; optional args])\nsample(model, AIS(N), MCMCThreads(), Ns, Nc[; optional keyword args])\nsample(model, AIS(N), MCMCDistributed(), Ns, Nc[; optional keyword args])\n\nGeneralities\n\nThis function will run an Affine Invariant MCMC sampler, and will return a Particles object for each parameter, the mandatory parameters are:\n\nmodel: a subtype of AbstractDensity, look at ApproxPosterior, ApproxKernelizedPosterior, CommonLogDensity.\n\nN: number of particles in the ensemble, this particles will be evolved to generate new samples.\n\nNs: total number of samples which must be recorded.\n\nNc: total number of chains to run in parallel if MCMCThreads or MCMCDistributed is enabled.\n\nthe optional arguments available are:\n\ndiscard_initial: number of mcmc particles to discard before saving any sample.\n\nntransitions: number of mcmc steps per particle between each sample.\n\nretry_sampling: number of maximum attempts to resample an initial particle whose cost (or log-density) is ±∞ or NaN.\n\nprogress: a boolean to disable verbosity\n\nMinimal Example for CommonLogDensity:\n\nusing KissABC\nD = CommonLogDensity(\n    2, #number of parameters\n    rng -> randn(rng, 2), # initial sampling strategy\n    x -> -100 * (x[1] - x[2]^2)^2 - (x[2] - 1)^2, # rosenbrock banana log-density\n)\nres = sample(D, AIS(50), 1000, ntransitions = 100, discard_initial = 500, progress = false)\nprintln(res)\n\noutput:\n\nParticles{Float64,1000}[1.43 ± 1.4, 0.99 ± 0.67]\n\nMinimal Example for ApproxKernelizedPosterior (ApproxPosterior)\n\nusing KissABC\nprior = Uniform(-10, 10) # prior distribution for parameter\nsim(μ) = μ + rand((randn() * 0.1, randn())) # simulator function\ncost(x) = abs(sim(x) - 0.0) # cost function to compare simulations to target data, in this case simply '0'\nplan = ApproxPosterior(prior, cost, 0.01) # Approximate model of log-posterior density (ABC)\n#                                           ApproxKernelizedPosterior can be used in the same fashion\nres = sample(plan, AIS(100), 2000, discard_initial = 10000, progress = false)\nprintln(res)\n\noutput:\n\n0.0 ± 0.46\n\n\n\n\n\n","category":"function"}]
}
