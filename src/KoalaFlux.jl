module KoalaFlux 

# new:
export FluxRegressor 
export CategoricalEmbedder

# needed in this module:
import Koala: Regressor, Classifier, softwarn, BaseType, rms
import Koala: supports_ensembling, Machine, SupervisedMachine
import DataFrames: eltypes, AbstractDataFrame, head
import KoalaTransforms: Transformer, TransformerMachine, ChainTransformer
import KoalaTransforms: MakeCategoricalsIntTransformer, Standardizer
import KoalaTransforms: RegressionTargetTransformer
import KoalaTransforms: ToIntScheme, ToIntTransformer
import Flux
import StatsBase

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: default_transformer_X, default_transformer_y, clean!
import Koala: transform, inverse_transform

# Notes on internals of this module:

# 1. What Flux.jl calls a "model" is not a model in the Koala.jl
# sense. Since we require "models" in the Flux.jl sense to be wrapped
# in `Flux.Chain` objects, we use the word "chain" in place of "Flux
# model" below. Each "chain", then, is a neural network (including
# weights).

# 2. Because a `FluxRegressor` machine allows for nominal input
# features, a FluxRegressor machine includes a chain (neural network)
# *and* a vector of matrices ("tracked" in the sense of Flux.jl)
# encoding the embedding, called "embedding" below. Thus a Koala
# `predictor` consists of a (chain, embedding) tuple.


## HELPERS

A ⊂ B = issubset(A, B)

Base.isempty(chain::Flux.Chain) = isempty(chain.layers)

default_formula(n_classes; threshold=10) =
    min(round(Int, sqrt(n_classes)), threshold)

function default_creator(n_inputs)
    n_hidden = round(Int, sqrt(n_inputs))
    return Flux.Chain(Flux.Dense(n_inputs, n_hidden, Flux.σ),
                      Flux.Dense(n_hidden, 1))
end

function ordinal_nominal_features(X::AbstractDataFrame)

    features = names(X)
    types = eltypes(X)
    ordinal_features = Symbol[]
    nominal_features = Symbol[]
    for j in eachindex(features)
        if types[j] <: AbstractFloat
            push!(ordinal_features, features[j])
        else
            push!(nominal_features, features[j])
        end
    end

    return ordinal_features, nominal_features

end

        
## MODEL TYPE DEFINITIONS

# type of matrix for embedding a single nominal feature:
EmbType = Flux.TrackedArray{Float64,2,Array{Float64,2}}

FluxPredictorType = Tuple{Vector{EmbType},Flux.Chain}

mutable struct FluxRegressor <: Regressor{FluxPredictorType}
    network_creator::Function
    dimension_formula::Function
    learning_rate::Float64
    lambda::Float64
    alpha::Float64 
    n::Int
end

# keyword constructor:
function FluxRegressor(; network_creator=default_creator,
                       dimension_formula=default_formula,
                       lambda = 1e-5, alpha=0.0, learning_rate=0.001, n=20)
    model = FluxRegressor(network_creator, dimension_formula, learning_rate, lambda, alpha, n)
    softwarn(clean!(model)) 
    return model
end

function clean!(model::FluxRegressor)
    message = ""
    if model.alpha > 1 || model.alpha < 0
        message = message*"alpha must be in range [0, 1]. Resetting to 0.0. "
        model.alpha = 0.0
    end
    return message
end


## TRANSFORMERS

struct FluxInput <: BaseType
    ordinal::Matrix{Float64}
    nominal::Matrix{Int}
end

function Base.getindex(X::KoalaFlux.FluxInput, rows::AbstractVector{Int}, ::Colon)
    ordinal = isempty(X.ordinal) ? Array{Float64}(0, length(rows)) :
        getindex(X.ordinal, :, rows)
    nominal = isempty(X.nominal) ? Array{Int}(0, length(rows)) :
        getindex(X.nominal, :, rows)
    return FluxInput(ordinal, nominal)
end

struct FrameToFluxInputTransformerScheme <: BaseType
    make_categoricals_int_transformer_machine::TransformerMachine
    ordinal_features::Vector{Symbol}
    nominal_features::Vector{Symbol}
end    

struct FrameToFluxInputTransformer <: Transformer
end

function fit(transformer::FrameToFluxInputTransformer, X::AbstractDataFrame,
             parallel, verbosity)

    make_categoricals_int_tansformer_machine =
        Machine(MakeCategoricalsIntTransformer(sorted=true), X)
    
    ordinal_features, nominal_features = ordinal_nominal_features(X)

    return FrameToFluxInputTransformerScheme(
        make_categoricals_int_tansformer_machine,
        ordinal_features, nominal_features)

end

function transform(transformer::FrameToFluxInputTransformer,
                   scheme, X::AbstractDataFrame)

    X = transform(scheme.make_categoricals_int_transformer_machine, X)
    ordinal_features, nominal_features = ordinal_nominal_features(X)

    # check X has same features as X used to fit:
    Set(ordinal_features) == Set(scheme.ordinal_features) &&
        Set(nominal_features) == Set(scheme.nominal_features) ||
        throw(DimensionMismatch("Missing or unexpected features encountered."))

    X_ordinal = X[scheme.ordinal_features]
    X_nominal = X[scheme.nominal_features]

    ordinal = isempty(X_ordinal) ? Array{Float64}(size(X, 2), 0) :
        transpose(Array(X_ordinal))
    nominal = isempty(X_nominal) ? Array{Int}(size(X, 2), 0) :
        transpose(Array(X_nominal))

    return FluxInput(ordinal, nominal)

end

default_transformer_X(model::FluxRegressor) = ChainTransformer(FrameToFluxInputTransformer(), Standardizer())
default_transformer_y(model::FluxRegressor) = RegressionTargetTransformer()


## TRAINING AND PREDICTION METHODS

mutable struct FluxCache <: BaseType
    X::FluxInput
    y::Vector{Float64}
    ordinal_features::Vector{Symbol}
    nominal_features::Vector{Symbol}
    make_categoricals_int_transformer_machine::TransformerMachine
    embedding::Vector{EmbType}
    chain::Flux.Chain
    FluxCache(X, y, ordinal_features, nominal_features, machine) =
        new(X, y, ordinal_features, nominal_features, machine)
end

setup(model::FluxRegressor, Xt, yt, scheme_X, parallel, verbosity) =
    FluxCache(Xt, yt, scheme_X[end].ordinal_features, scheme_X[end].nominal_features,
              scheme_X[end].make_categoricals_int_transformer_machine)


# function to concatenate vector of ordinal values with vector of
# embedded nominal values into single vector:
function x(embedding, ordinals, nominals)
    embedded_nominals = vcat([embedding[j][:,nominals[j]]
                              for j in eachindex(embedding)]...)
    return vcat(ordinals, embedded_nominals)
end    

function fit(model::FluxRegressor, cache, add, parallel, verbosity; args...)

    if !add

        # create new embedding matrices:
        cache.embedding = Array{EmbType}(length(cache.nominal_features))
        schemes=cache.make_categoricals_int_transformer_machine.scheme.schemes
        dimensions = Int[]
        if verbosity >= 0
            info("")
            info("Categorical feature embeddings:")
            info("  feature     \t| embedding dimension")
            info("------------------|----------------------")
        end
        
        for j in eachindex(cache.nominal_features)
            n_levels = schemes[j].n_levels
            d = model.dimension_formula(n_levels)
            append!(dimensions, d)
            cache.embedding[j] = Flux.param(randn(d, n_levels))
            verbosity < 1 || info("  $(cache.nominal_features[j]) \t| $d")
        end

        # create a new post-embedding network:
        n_inputs = length(cache.ordinal_features) + sum(dimensions)
        cache.chain = model.network_creator(n_inputs)
        verbosity < 1 || info("Flux network architecture: \n  $(cache.chain)")
        
    end

    Flux.testmode!(cache.chain, false)
    
    chain_params = Flux.params(cache.chain)

    ntrain = length(cache.y)
    train = 1:ntrain

    verbosity < 1 || println()
    
    for eon in 1:model.n
        verbosity < 1 || print("\rTraining eon number: $eon   ")
        scrambled_train = StatsBase.sample(train, ntrain, replace=false)
        for i in scrambled_train
            ordinals = cache.X.ordinal[:,i]
            nominals = cache.X.nominal[:,i]
            input_vector = x(cache.embedding, ordinals, nominals)
            l = rms(cache.chain(input_vector), cache.y[i])
            l2_penalty = sum(vecnorm, chain_params)
            l1_penalty = sum(x->vecnorm(x, 1), chain_params)
            if !isempty(cache.embedding)
                l2_penalty += sum(vecnorm, cache.embedding)
                l1_penalty += sum(x->vecnorm(x, 1), cache.embedding)
            end
            l += model.lambda*(model.alpha*l1_penalty + (1 - model.alpha)*l2_penalty)
            # some loss functions are not differentiable at zero, so:
            if abs(l) > eps(Float64)
                Flux.back!(l)
                for p in cache.embedding
                    Flux.Tracker.update!(p, -model.learning_rate*Flux.Tracker.grad(p))
                end
                for p in chain_params
                    Flux.Tracker.update!(p, -model.learning_rate*Flux.Tracker.grad(p))
                end
            end
        end
    end

    report = Dict{Symbol,Any}()
    report[:embedding] = cache.embedding

    predictor = (cache.embedding, cache.chain)

    return predictor, report, cache

end
    
function predict(model::FluxRegressor, predictor::FluxPredictorType,
                 X, parallel, verbosity)

    embedding, chain = predictor
    Flux.testmode!(chain)
    npatterns = size(X.ordinal, 2) 

    predictions = Array{Float64}(npatterns)
    for i in 1:npatterns
        ordinals = X.ordinal[:,i]
        nominals = X.nominal[:,i]
        input_vector = x(embedding, ordinals, nominals)
        predictions[i] = chain(input_vector).data[1]
    end

    return predictions
end


## FEATURE EMBEDDING TRANSFORMS

# Here we define transforms to embed the categorical features in some
# DataFrame into the multidemsional continous variables learned by a
# Flux neural network.

mutable struct CategoricalEmbedder <: Transformer
    flux_machine::SupervisedMachine{FluxPredictorType, FluxRegressor}
    features::Vector{Symbol}
end

CategoricalEmbedder(machine::SupervisedMachine{FluxPredictorType, FluxRegressor};
                    features = Symbol[]) = CategoricalEmbedder(machine, features)
CategoricalEmbedder() =
    error("You must give `CategoricalEmbedder` a `FluxMachine` argument to instantiate it.")

struct CategoricalEmbedderScheme <: BaseType
    features::Vector{Symbol}
    to_int_transformer::ToIntTransformer
    to_int_scheme_given_feature::Dict{Symbol,ToIntScheme}
    embedding_given_feature::Dict{Symbol,Matrix{Float64}}
end

function fit(transformer::CategoricalEmbedder,
             X::AbstractDataFrame, parallel, verbosity)

    _, nominal = ordinal_nominal_features(X)

    if isempty(transformer.features)
        features = nominal
    else
        features = transformer.features
    end
    
    flux_nominals = transformer.flux_machine.scheme_X[end].nominal_features
    Set(features) ⊂ Set(flux_nominals) ||
        throw(DimensionMismatch("Encountered categorical "*
                                "not appearing in `FluxRegressor` object."))

    
    to_int_scheme_given_feature = Dict{Symbol,ToIntScheme}()
    embedding_given_feature = Dict{Symbol,Matrix{Float64}}()

    machine = transformer.flux_machine
    scheme = machine.scheme_X[end]
    to_int_schemes =
        scheme.make_categoricals_int_transformer_machine.scheme.schemes
    to_int_transformer =
        scheme.make_categoricals_int_transformer_machine.scheme.to_int_transformer
    
    for j in eachindex(flux_nominals)
        if flux_nominals[j] in features
            ftr = flux_nominals[j]
            to_int_scheme_given_feature[ftr] =
                to_int_schemes[j]
            embedding_given_feature[ftr] =
                machine.report[:embedding][j].data
        end
    end
    
    return CategoricalEmbedderScheme(features, to_int_transformer,
                                     to_int_scheme_given_feature,
                                     embedding_given_feature)
end

# Will use the following to get the `kth` component of the emedding
# vector for raw value `x` of feature `ftr`. Will work just as well if
# `x` is replaced with a vector of categorical values, returning a
# Float64 vector.
function transform(scheme::CategoricalEmbedderScheme, ftr::Symbol, k::Int, x)
    x_as_int = transform(scheme.to_int_transformer,
                         scheme.to_int_scheme_given_feature[ftr], x)
    return scheme.embedding_given_feature[ftr][k,x_as_int]
end

function transform(transformer::CategoricalEmbedder, scheme, X::AbstractDataFrame)
    
    _, nominals = ordinal_nominal_features(X)
    without_embedding = setdiff(Set(nominals), Set(scheme.features))
    if !isempty(without_embedding) 
        warn("No embeddings found for the following categorical features: ")
        for ftr in without_embedding
            println(ftr)
        end
    end

    Xout = X[:,:] # copy of X, working even if X is subdataframe
    for ftr in intersect(Set(nominals),Set(scheme.features))
        n_levels = scheme.to_int_scheme_given_feature[ftr].n_levels
        for k in 1:n_levels
            subftr = Symbol(string(ftr,"__",k))
            # in the (rare) case subft is not a new feature label:
            while subftr in names(Xout)
                subftr = Symbol(string(subftr,"_"))
            end
            Xout[subftr] = transform(scheme, ftr, k, X[ftr])
        end
        delete!(Xout, ftr)
    end

    return Xout

end


end # of module

