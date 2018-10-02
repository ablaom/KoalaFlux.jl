module KoalaFlux 

# new:
export FluxRegressor 
export CategoricalEmbedder

# needed in this module:
using LinearAlgebra
using Random
import Koala: Regressor, Classifier, softwarn, BaseType, rms
import Koala: supports_ensembling, Machine, SupervisedMachine
import DataFrames: eltypes, AbstractDataFrame, head
import KoalaTransforms: Transformer, TransformerMachine, ChainTransformer
import KoalaTransforms: MakeCategoricalsIntTransformer, Standardizer
import KoalaTransforms: RegressionTargetTransformer
import KoalaTransforms: ToIntScheme, ToIntTransformer
import Flux

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

# 2. Because a `FluxRegressor` machine allows for categorical input
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

function ordinal_categorical_features(X::AbstractDataFrame)

    features = names(X)
    types = eltypes(X)
    ordinal_features = Symbol[]
    categorical_features = Symbol[]
    for j in eachindex(features)
        if types[j] <: AbstractFloat
            push!(ordinal_features, features[j])
        else
            push!(categorical_features, features[j])
        end
    end

    return ordinal_features, categorical_features

end

        
## MODEL TYPE DEFINITIONS

# type of matrix for embedding a single categorical feature:
EmbType = Flux.TrackedArray{Float64,2,Array{Float64,2}}
WeightsType = Union{Flux.TrackedArray{Float64,2,Array{Float64,2}},
                       Flux.TrackedArray{Float64,1,Array{Float64,1}}}

FluxPredictorType = Tuple{Vector{EmbType},Flux.Chain}

mutable struct FluxRegressor <: Regressor{FluxPredictorType}
    network_creator::Function
    dimension_formula::Function
    learning_rate::Float64
    inertia::Float64
    lambda::Float64
    alpha::Float64
    embedding0::Vector{EmbType}
    chain0::Flux.Chain
    n::Int
end

# keyword constructor:
function FluxRegressor(; network_creator=default_creator,
                       dimension_formula=default_formula,
                       learning_rate=0.001,
                       inertia=0.9, lambda = 1e-5, alpha=0.0,
                       embedding0=EmbType[], chain0=Flux.Chain(), n=20)
    model = FluxRegressor(network_creator, dimension_formula, learning_rate,
                          inertia, lambda, alpha, embedding0, chain0, n)
    softwarn(clean!(model)) 
    return model
end

function clean!(model::FluxRegressor)
    message = ""
    if model.alpha > 1 || model.alpha < 0
        message = message*"alpha must be in range [0, 1]. Resetting to 0.0. "
        model.alpha = 0.0
    end
    if model.inertia > 1 || model.inertia < 0
        message = message*"inertia must be in range [0, 1]. Resetting to 0.0. "
        model.inertia = 0.0
    end
    return message
end


## TRANSFORMERS

struct FluxInput <: BaseType
    ordinal::Matrix{Float64}
    categorical::Matrix{Int}
end

function Base.getindex(X::KoalaFlux.FluxInput, rows::AbstractVector{Int}, ::Colon)
    ordinal = isempty(X.ordinal) ? Array{Float64}(undef, 0, length(rows)) :
        getindex(X.ordinal, :, rows)
    categorical = isempty(X.categorical) ? Array{Int}(undef, 0, length(rows)) :
        getindex(X.categorical, :, rows)
    return FluxInput(ordinal, categorical)
end

struct FrameToFluxInputTransformerScheme <: BaseType
    make_categoricals_int_transformer_machine::TransformerMachine
    ordinal_features::Vector{Symbol}
    categorical_features::Vector{Symbol}
end    

struct FrameToFluxInputTransformer <: Transformer
end

function fit(transformer::FrameToFluxInputTransformer, X::AbstractDataFrame,
             parallel, verbosity)

    make_categoricals_int_tansformer_machine =
        Machine(MakeCategoricalsIntTransformer(sorted=true), X)
    
    ordinal_features, categorical_features = ordinal_categorical_features(X)

    return FrameToFluxInputTransformerScheme(
        make_categoricals_int_tansformer_machine,
        ordinal_features, categorical_features)

end

function transform(transformer::FrameToFluxInputTransformer,
                   scheme, X::AbstractDataFrame)

    X = transform(scheme.make_categoricals_int_transformer_machine, X)
    ordinal_features, categorical_features = ordinal_categorical_features(X)

    # check X has same features as X used to fit:
    Set(ordinal_features) == Set(scheme.ordinal_features) &&
        Set(categorical_features) == Set(scheme.categorical_features) ||
        throw(DimensionMismatch("Missing or unexpected features encountered."))

    X_ordinal = X[scheme.ordinal_features]
    X_categorical = X[scheme.categorical_features]

    ordinal = isempty(X_ordinal) ? Array{Float64}(undef, size(X, 2), 0) :
        transpose(convert(Array, X_ordinal))
    categorical = isempty(X_categorical) ? Array{Int}(undef, size(X, 2), 0) :
        transpose(convert(Array, X_categorical))

    return FluxInput(ordinal, categorical)

end

default_transformer_X(model::FluxRegressor) = ChainTransformer(FrameToFluxInputTransformer(), Standardizer())
default_transformer_y(model::FluxRegressor) = RegressionTargetTransformer()


## TRAINING AND PREDICTION METHODS

mutable struct FluxCache <: BaseType
    X::FluxInput
    y::Vector{Float64}
    ordinal_features::Vector{Symbol}
    categorical_features::Vector{Symbol}
    make_categoricals_int_transformer_machine::TransformerMachine
    embedding::Vector{EmbType}
    chain::Flux.Chain
    FluxCache(X, y, ordinal_features, categorical_features, machine) =
        new(X, y, ordinal_features, categorical_features, machine)
end

setup(model::FluxRegressor, Xt, yt, scheme_X, parallel, verbosity) =
    FluxCache(Xt, yt, scheme_X[end].ordinal_features, scheme_X[end].categorical_features,
              scheme_X[end].make_categoricals_int_transformer_machine)


# function to concatenate vector of ordinal values with vector of
# embedded categorical values into single vector:
function x(embedding, ordinals, categoricals)
    embedded_categoricals = vcat([embedding[j][:,categoricals[j]]
                              for j in eachindex(embedding)]...)
    return vcat(ordinals, embedded_categoricals)
end    

function fit(model::FluxRegressor, cache, add, parallel, verbosity; args...)

    if !add

        # create or load initial embedding matrices:
        if isempty(model.embedding0)
            cache.embedding = Array{EmbType}(undef, length(cache.categorical_features))
        else
            cache.embedding = model.embedding0
        end
        
        # print embedding details:
        schemes=cache.make_categoricals_int_transformer_machine.scheme.schemes
        dimensions = Int[]
        if verbosity >= 1
            @info "\n"
            @info "Categorical feature embeddings:"
            @info "  feature     \t| embedding dimension"
            @info "------------------|----------------------"
        end
        for j in eachindex(cache.categorical_features)
            n_levels = schemes[j].n_levels
            d = model.dimension_formula(n_levels)
            append!(dimensions, d)
            cache.embedding[j] = Flux.param(randn(d, n_levels))
            verbosity < 1 || @info "  $(cache.categorical_features[j]) \t| $d"
        end

        # create or load a post-embedding network:
        if isempty(model.chain0)
            n_inputs = length(cache.ordinal_features) + sum(dimensions)
            cache.chain = model.network_creator(n_inputs)
        else
            cache.chain = model.chain0
        end
        verbosity < 1 || @info "Flux network architecture: \n  $(cache.chain)"
        
    end

    Flux.testmode!(cache.chain, false)
    
    embed_params = cache.embedding
    chain_params = Flux.params(cache.chain)

    ntrain = length(cache.y)
    train = 1:ntrain

    verbosity < 1 || println()

    # Note: In our terminology a "momentum" is a "moving average"
    # estimate of gradient.

    # initialize vectors of momenta:
    embed_momenta = Array{Array{Float64,2}}(undef, length(embed_params))
    chain_momenta = Array{Union{Array{Float64,1},
                                Array{Float64,2}}}(undef, length(chain_params))
    for k in eachindex(embed_params)
        p = embed_params[k]
        embed_momenta[k] = zeros(size(p))
    end
    for k in eachindex(chain_params)
        p = chain_params[k]
        chain_momenta[k] = zeros(size(p))
    end
    
    η = model.learning_rate
    β = model.inertia

    for eon in 1:model.n
        verbosity < 1 || print("\rTraining eon number: $eon   ")
        scrambled_train = shuffle(train)
        for i in scrambled_train
            ordinals = cache.X.ordinal[:,i]
            categoricals = cache.X.categorical[:,i]
            input_vector = x(embed_params, ordinals, categoricals)
#            @show input_vector
            l = rms(cache.chain(input_vector), cache.y[i])
#            @show cache.y[i]
#            @show chain_params
#            @show cache.chain(input_vector)
#            @show l
            l2_penalty = sum(norm, chain_params)
#            @show l2_penalty
            l1_penalty = sum(x->norm(x, 1), chain_params)
#            @show l1_penalty
            if !isempty(embed_params)
                l2_penalty += sum(norm, embed_params)
                l1_penalty += sum(x->norm(x, 1), embed_params)
            end
            l += model.lambda*(model.alpha*l1_penalty + (1 - model.alpha)*l2_penalty)
            !isnan(l) || @error "A NaN loss encountered."
            # some loss functions are not differentiable at zero, so:
            if abs(l) > eps(Float64)
#                @show l
                Flux.back!(l)
                for k in eachindex(embed_params)
                    p = embed_params[k]
                    embed_momenta[k] =  β*embed_momenta[k] +
                        (1 - β)*Flux.Tracker.grad(p)
                    Flux.Tracker.update!(p, -η*embed_momenta[k])
                end
                for k in eachindex(chain_params)
                    p = chain_params[k]
                    chain_momenta[k] =  β*chain_momenta[k] +
                        (1 - β)*Flux.Tracker.grad(p)
                    Flux.Tracker.update!(p, -η*chain_momenta[k])
                end
            end
        end
    end

    report = Dict{Symbol,Any}()
    report[:embedding] = embed_params
    report[:chain] = cache.chain

    predictor = (embed_params, cache.chain)

    return predictor, report, cache

end
    
function predict(model::FluxRegressor, predictor::FluxPredictorType,
                 X, parallel, verbosity)

    embedding, chain = predictor
    Flux.testmode!(chain)
    npatterns = size(X.ordinal, 2) 

    predictions = Array{Float64}(undef, npatterns)
    for i in 1:npatterns
        ordinals = X.ordinal[:,i]
        categoricals = X.categorical[:,i]
        input_vector = x(embedding, ordinals, categoricals)
        predictions[i] = chain(input_vector).data[1]
    end

    return predictions
end


## FEATURE EMBEDDING TRANSFORMS

# Here we define transforms to embed the categorical features in some
# DataFrame into the multidimensional continuous variables learned by a
# Flux neural network.

mutable struct CategoricalEmbedder <: Transformer
    flux_machine::SupervisedMachine{FluxPredictorType, FluxRegressor}
    features::Vector{Symbol}
end

CategoricalEmbedder(machine::SupervisedMachine{FluxPredictorType, FluxRegressor};
                    features = Symbol[]) = CategoricalEmbedder(machine, features)
CategoricalEmbedder() =
    @error "You must give `CategoricalEmbedder` a `FluxMachine` 
         argument to instantiate it."

struct CategoricalEmbedderScheme <: BaseType
    features::Vector{Symbol}
    to_int_transformer::ToIntTransformer
    to_int_scheme_given_feature::Dict{Symbol,ToIntScheme}
    embedding_given_feature::Dict{Symbol,Matrix{Float64}}
end

function fit(transformer::CategoricalEmbedder,
             X::AbstractDataFrame, parallel, verbosity)

    _, categorical = ordinal_categorical_features(X)

    if isempty(transformer.features)
        features = categorical
    else
        features = transformer.features
    end
    
    flux_categoricals = transformer.flux_machine.scheme_X[end].categorical_features
    Set(features) ⊂ Set(flux_categoricals) ||
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
    
    for j in eachindex(flux_categoricals)
        if flux_categoricals[j] in features
            ftr = flux_categoricals[j]
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
    
    _, categoricals = ordinal_categorical_features(X)
    without_embedding = setdiff(Set(categoricals), Set(scheme.features))
    if !isempty(without_embedding) 
        @warn "No embeddings found for the following categorical features: "
        for ftr in without_embedding
            println(ftr)
        end
    end

    Xout = X[:,:] # copy of X, working even if X is subdataframe
    for ftr in intersect(Set(categoricals),Set(scheme.features))
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

