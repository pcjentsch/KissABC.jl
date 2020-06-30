var documenterSearchIndex = {"docs":
[{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"EditURL = \"https://github.com/JuliaApproxInference/KissABC.jl/blob/master/docs/literate/example_1.jl\"","category":"page"},{"location":"example_1/#A-gaussian-mixture-model-1","page":"Example: Gaussian Mixture","title":"A gaussian mixture model","text":"","category":"section"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"First of all we define our model,","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"using KissABC\nusing Distributions\n\nfunction model(P, N)\n    μ_1, μ_2, σ_1, σ_2, prob = P\n    d1 = randn(N) .* σ_1 .+ μ_1\n    d2 = randn(N) .* σ_2 .+ μ_2\n    ps = rand(N) .< prob\n    R = zeros(N)\n    R[ps] .= d1[ps]\n    R[.!ps] .= d2[.!ps]\n    R\nend","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"Let's use the model to generate some data, this data will constitute our dataset","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"parameters = (1.0, 0.0, 0.2, 2.0, 0.4)\ndata = model(parameters, 5000)","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"let's look at the data","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"using Plots\nhistogram(data)\nsavefig(\"ex1_hist1.svg\");\nnothing; # hide\nnothing #hide","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"(Image: ex1_hist1)","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"we can now try to infer all parameters using KissABC, first of all we need to define a reasonable prior for our model","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"prior = Factored(\n    Uniform(0, 2), # there is surely a peak between 0 and 2\n    Uniform(-1, 1), #there is a smeared distribution centered around 0\n    Uniform(0, 1), # the peak has surely a width below 1\n    Uniform(0, 4), # the smeared distribution surely has a width less than 4\n    Beta(2, 2), # the number of total events from both distributions look about the same, so we will favor 0.5 just a bit\n);\nnothing #hide","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"let's look at a sample from the prior, to see that it works","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"rand(prior)","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"now we need a function to compute summary statistics for our data, this is not the optimal choice, but it will work out anyway","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"function S(x)\n    r = (0.1, 0.2, 0.5, 0.8, 0.9)\n    quantile(x, r)\nend","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"we will define a function to use the model and summarize it's results","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"summ_model(P, N) = S(model(P, N));\nnothing #hide","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"now we need a distance function to compare the summary statistics of target data and simulated data","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"summ_data = S(data)\nD(P, N = 5000) = sqrt(mean(abs2, summ_data .- summ_model(P, N)));\nnothing #hide","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"we can now run ABCDE to get the posterior distribution of our parameters given the dataset data","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"approx_density = ApproxPosterior(prior, D, 0.05)\nres, _ = mcmc(approx_density, nparticles = 100, generations = 500, verbose = 0)","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"let's see the median and 95% confidence interval for the inferred parameters and let's compare them with the true values","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"getstats(V) =\n    (median = median(V), lowerbound = quantile(V, 0.025), upperbound = quantile(V, 0.975));\n\nlabels = (:μ_1, :μ_2, :σ_1, :σ_2, :prob)\nP = [getindex.(res, i) for i = 1:5]\nstats = getstats.(P)\n\nfor is in eachindex(stats)\n    println(labels[is], \" ≡ \", parameters[is], \" → \", stats[is])\nend","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"The inferred parameters are close to nominal values","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"This page was generated using Literate.jl.","category":"page"},{"location":"reference/#","page":"Reference","title":"Reference","text":"CurrentModule = KissABC","category":"page"},{"location":"reference/#Reference-1","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/#","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"reference/#","page":"Reference","title":"Reference","text":"Modules = [KissABC]","category":"page"},{"location":"reference/#KissABC.ApproxKernelizedPosterior","page":"Reference","title":"KissABC.ApproxKernelizedPosterior","text":"ApproxPosterior(\n    prior::Distribution,\n    cost::Function,\n    max_cost::Real\n)\n\nthis function will return a type which can be used in the mcmc function as an ABC density, this type works by assuming uniformly distributed errors in [-ϵ,ϵ], ϵ is specified in the variable max_cost.\n\n\n\n\n\n","category":"type"},{"location":"reference/#KissABC.Factored","page":"Reference","title":"KissABC.Factored","text":"Factored{N} <: Distribution{Multivariate, MixedSupport}\n\na Distribution type that can be used to combine multiple UnivariateDistribution's and sample from them. Example: it can be used as prior = Factored(Normal(0,1), Uniform(-1,1))\n\n\n\n\n\n","category":"type"},{"location":"reference/#KissABC.mcmc-Tuple{KissABC.AbstractApproxDensity}","page":"Reference","title":"KissABC.mcmc","text":"function mcmc(\n    density::AbstractApproxDensity;\n    nparticles::Int,\n    generations,\n    rng::AbstractRNG = Random.GLOBAL_RNG,\n    parallel = false,\n    verbose = 2,\n)\n\nThis function will run an Affine Invariant MC sampler on the ABC density defined in density, the ensemble will contain nparticles particles, and each particle will evolve for a total number of steps equal to generations.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.length-Union{Tuple{Factored{N}}, Tuple{N}} where N","page":"Reference","title":"Base.length","text":"length(p::Factored) = begin\n\nreturns the number of distributions contained in p.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.rand-Union{Tuple{N}, Tuple{Random.AbstractRNG,Factored{N}}} where N","page":"Reference","title":"Base.rand","text":"rand(rng::AbstractRNG, factoreddist::Factored)\n\nfunction to sample one element from a Factored object\n\n\n\n\n\n","category":"method"},{"location":"reference/#Distributions.logpdf-Union{Tuple{N}, Tuple{Factored{N},Any}} where N","page":"Reference","title":"Distributions.logpdf","text":"logpdf(d::Factored, x) = begin\n\nFunction to evaluate the logpdf of a Factored distribution object\n\n\n\n\n\n","category":"method"},{"location":"reference/#Distributions.pdf-Union{Tuple{N}, Tuple{Factored{N},Any}} where N","page":"Reference","title":"Distributions.pdf","text":"pdf(d::Factored, x) = begin\n\nFunction to evaluate the pdf of a Factored distribution object\n\n\n\n\n\n","category":"method"},{"location":"reference/#KissABC.cdf_g_inv-Tuple{Any,Any}","page":"Reference","title":"KissABC.cdf_g_inv","text":"Inverse cdf of g-pdf, see eq. 10 of Foreman-Mackey et al. 2013.\n\n\n\n\n\n","category":"method"},{"location":"reference/#KissABC.sample_g-Tuple{Random.AbstractRNG,Any}","page":"Reference","title":"KissABC.sample_g","text":"Sample from g using inverse transform sampling.  a=2.0 is recommended.\n\n\n\n\n\n","category":"method"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"EditURL = \"https://github.com/JuliaApproxInference/KissABC.jl/blob/master/docs/literate/index.jl\"","category":"page"},{"location":"#KissABC-1","page":"Basic Usage","title":"KissABC","text":"","category":"section"},{"location":"#Usage-guide-1","page":"Basic Usage","title":"Usage guide","text":"","category":"section"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"The ingredients you need to use Approximate Bayesian Computation:","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"A simulation which depends on some parameters, able to generate datasets similar to your target dataset if parameters are tuned\nA prior distribution over such parameters\nA distance function to compare generated dataset to the true dataset","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"We will start with a simple example, we have a dataset generated according to an Normal distribution whose parameters are unknown","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"tdata = randn(1000) .* 0.04 .+ 2;\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"we are ofcourse able to simulate normal random numbers, so this constitutes our simulation","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"sim((μ, σ)) = randn(1000) .* σ .+ μ;\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"The second ingredient is a prior over the parameters μ and σ","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"using Distributions\nusing KissABC\nprior = Factored(Uniform(1, 3), Truncated(Normal(0, 0.1), 0, 100));\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"we have chosen a uniform distribution over the interval [1,3] for μ and a normal distribution truncated over ℝ⁺ for σ.","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"Now all that we need is a distance function to compare the true dataset to the simulated dataset, for this purpose comparing mean and std is optimal","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"function dist(x, y)\n    d1 = mean(x) - mean(y)\n    d2 = std(x) - std(y)\n    hypot(d1, d2 * 50)\nend","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"Now we are all set, we can use mcmc which is Affine Invariant MC algorithm, to simulate the posterior distribution for this model, inferring μ and σ","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"cost(x) = dist(tdata, sim(x))\napprox_density = ApproxPosterior(prior, cost, 0.1)\nres, _ =\n    mcmc(approx_density, nparticles = 2000, generations = 100, parallel = true, verbose = 0);\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"the parameters we chose are: a tolerance on distances equal to 0.1, a number of simulated particles equal to 2000 and total simulations per particle to 100, we enabled Threaded parallelism, the simulated posterior results are in res, while the _ is there to simply ignore all the other returned information. We can now extract the inference results:","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"prsample = [rand(prior) for i = 1:2000] #some samples from the prior for comparison\nμ_pr = getindex.(prsample, 1) # μ samples from the prior\nσ_pr = getindex.(prsample, 2) # σ samples from the prior\nμ_p = getindex.(res, 1) # μ samples from the posterior\nσ_p = getindex.(res, 2); # σ samples from the posterior\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"and plotting prior and posterior side by side we get:","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"using Plots\na = stephist(\n    μ_pr,\n    xlims = (1, 3),\n    xlabel = \"μ prior\",\n    leg = false,\n    lw = 2,\n    normalize = true,\n)\nb = stephist(\n    σ_pr,\n    xlims = (0, 0.3),\n    xlabel = \"σ prior\",\n    leg = false,\n    lw = 2,\n    normalize = true,\n)\nap = stephist(\n    μ_p,\n    xlims = (1, 3),\n    xlabel = \"μ posterior\",\n    leg = false,\n    lw = 2,\n    normalize = true,\n)\nbp = stephist(\n    σ_p,\n    xlims = (0, 0.3),\n    xlabel = \"σ posterior\",\n    leg = false,\n    lw = 2,\n    normalize = true,\n)\nplot(a, ap, b, bp)\nsavefig(\"inference.svg\");\nnothing; # hide\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"(Image: inference_plot)","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"we can see that the algorithm has correctly inferred both parameters, this exact recipe will work for much more complicated models and simulations, with some tuning.","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"This page was generated using Literate.jl.","category":"page"}]
}
