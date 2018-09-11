using Koala
using KoalaFlux
using Base.Test

X, y = load_ames()
@test KoalaFlux.ordinal_categorical_features(X) ==
    (Symbol[:OverallQual, :GrLivArea, :x1stFlrSF, :TotalBsmtSF,
            :BsmtFinSF1, :LotArea, :GarageCars, :GarageArea,
            :YearRemodAdd, :YearBuilt],
     Symbol[:Neighborhood, :MSSubClass])

t = KoalaFlux.FrameToFluxInputTransformer()
tM = Machine(t, X)
XX = transform(tM, X[2:6,:])
@test XX.categorical[2,1] ==  XX.categorical[2,3]
@test XX.categorical[1,1] != XX.categorical[1,2]


for ftr in tM.scheme.categorical_features
    delete!(X, ftr)
end
t = KoalaFlux.FrameToFluxInputTransformer()
tM = Machine(t, X)
XX = transform(tM, X[2:6,:])

# 3D embedding for a two-class categorical:
A1 = Float64[1 2;
             3 4;
             0 5]

# 2D embedding for a three-class categorical:
A2 = Float64[1 2 3;
             0 1 5]

embedding = [A1, A2]
@test KoalaFlux.x(embedding, [1.2, 2.3, 4.3, 5.6], [2, 1]) ==
    [1.2, 2.3, 4.3, 5.6, 2.0, 4.0, 5.0, 1.0, 0.0]

X, y = load_ames()
train, test = partition(eachindex(y), 0.7)
flux = FluxRegressor()
flux.n = 3
flux.learning_rate = 0.002
fluxM = Machine(flux, X, y, train)
fit!(fluxM, train)

function get_creator(p)
    function creator(n_inputs)
        n_hidden = round(Int, sqrt(n_inputs))
        return Flux.Chain(Flux.Dense(n_inputs, n_hidden, Flux.σ),
                          Flux.Dropout(p), Flux.Dense(n_hidden, 1))
    end
    return creator
end

flux.network_creator = get_creator(0.5)
learning_curve(fluxM, train, test, [2,4,6])

flux.embedding0 = fluxM.report[:embedding]
flux.chain0 = fluxM.report[:chain]
fit!(fluxM)


# tests for CategoricalEmbedder:

letters = ['a', 'b', 'a']
weirds = ["goo", "goo", "prickles"]
numbers = [1.0, 2.0, 1.0]
yy = [-1.0, 0.0, 1.0]

XX = DataFrames.DataFrame(letters=letters, numbers=numbers, weirds=weirds)
flux.n = 1
flux.dimension_formula = x -> x
flux.learning_rate = 0.0
fluxM = Machine(flux, XX, yy, 1:3)
fit!(fluxM, 1:3)
afloat, bfloat = fluxM.report[:embedding][1].data[1,:]
t = CategoricalEmbedder(fluxM)
tM = Machine(t, XX)
s = tM.scheme
@test transform(s, :letters, 1, letters) ≈ [afloat, bfloat, afloat]
transform(tM, XX)
@test transform(tM, XX[[:letters]])[:letters__1] == transform(tM, XX)[:letters__1]
