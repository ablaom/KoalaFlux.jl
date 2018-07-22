module KoalaFlux 

# new:
export FluxRegressor # eg, RidgeRegressor

# needed in this module:
import Koala: Regressor, Classifier, softwarn, BaseType, rms
import Koala: supports_ensembling, Machine
import DataFrames: eltypes, AbstractDataFrame, head
import KoalaTransforms: Transformer, TransformerMachine
import KoalaTransforms: MakeCategoricalsIntTransformer, Standardizer
import KoalaTransforms: RegressionTargetTransformer
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

# supports_ensembling(model::FluxRegressor) = false

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
    η::Float64
    n::Int
end

# keyword constructor:
function FluxRegressor(; network_creator=default_creator,
                       dimension_formula=default_formula,
                       η=0.03, n=1)
    model = FluxRegressor(network_creator, dimension_formula, η, n)
    softwarn(clean!(model)) 
    return model
end

clean!(model::FluxRegressor) = ""


## TRANSFORMERS

struct FluxInput <: BaseType
    ordinal::Matrix{Float64}
    nominal::Matrix{Int}
end

Base.getindex(X::KoalaFlux.FluxInput, rows::AbstractVector{Int}, ::Colon) =
    FluxInput(getindex(X.ordinal, :, rows), getindex(X.nominal, :, rows))

struct FrameToFluxInputTransformerScheme <: BaseType
    make_categoricals_int_transformer_machine::TransformerMachine
    standardizer_machine::TransformerMachine
    ordinal_features::Vector{Symbol}
    nominal_features::Vector{Symbol}
end    

struct FrameToFluxInputTransformer <: Transformer
end

function fit(transformer::FrameToFluxInputTransformer, X::AbstractDataFrame,
             parallel, verbosity)

    make_categoricals_int_tansformer_machine =
        Machine(MakeCategoricalsIntTransformer(sorted=true), X)
    
    standardizer_machine =
        Machine(Standardizer(), X)
    
    ordinal_features, nominal_features = ordinal_nominal_features(X)

    return FrameToFluxInputTransformerScheme(
        make_categoricals_int_tansformer_machine,
        standardizer_machine,
        ordinal_features, nominal_features)

end

function transform(transformer::FrameToFluxInputTransformer,
                   scheme, X::AbstractDataFrame)

    X = transform(scheme.make_categoricals_int_transformer_machine, X)
    X = transform(scheme.standardizer_machine, X)
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

default_transformer_X(model::FluxRegressor) = FrameToFluxInputTransformer()
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
    FluxCache(Xt, yt, scheme_X.ordinal_features, scheme_X.nominal_features,
              scheme_X.make_categoricals_int_transformer_machine)


# function to concatenate vector of ordinal values with vector of
# embedded nominal values into single vector:
function x(embedding, ordinals, nominals)
    embedded_nominals = vcat([embedding[j][:,nominals[j]] for j in eachindex(embedding)]...)
    return vcat(ordinals, embedded_nominals)
end    

function fit(model::FluxRegressor, cache, add, parallel, verbosity; args...)

    if !add

        # create new embedding matrices:
        cache.embedding = Array{EmbType}(length(cache.nominal_features))
        schemes=cache.make_categoricals_int_transformer_machine.scheme.schemes
        dimensions = Int[]
        for j in eachindex(cache.nominal_features)
            n_levels = schemes[j].n_levels
            d = model.dimension_formula(n_levels)
            append!(dimensions, d)
            cache.embedding[j] = Flux.param(randn(d, n_levels))
        end

        # create a new post-embedding network:
        n_inputs = length(cache.ordinal_features) + sum(dimensions)
        cache.chain = model.network_creator(n_inputs)
        
    end

    chain_params = Flux.params(cache.chain)

    ntrain = length(cache.y)
    train = 1:ntrain

    for eon in 1:model.n
        
        scrambled_train = StatsBase.sample(train, ntrain, replace=false)
        count = 0
        sum = 0.0
        for i in scrambled_train
            ordinals = cache.X.ordinal[:,i]
            nominals = cache.X.nominal[:,i]
            input_vector = x(cache.embedding, ordinals, nominals)
            l = rms(cache.chain(input_vector), cache.y[i])
            Flux.back!(l)
            for p in cache.embedding
                Flux.Tracker.update!(p, -model.η*Flux.Tracker.grad(p))
            end
            for p in chain_params
                Flux.Tracker.update!(p, -model.η*Flux.Tracker.grad(p))
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


        
end # of module

