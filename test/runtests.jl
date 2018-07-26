# addprocs()
# using Revise
using Koala
using KoalaFlux
using Base.Test

X, y = load_ames()
@test KoalaFlux.ordinal_nominal_features(X) ==
    (Symbol[:OverallQual, :GrLivArea, :x1stFlrSF, :TotalBsmtSF,
            :BsmtFinSF1, :LotArea, :GarageCars, :GarageArea,
            :YearRemodAdd, :YearBuilt],
     Symbol[:Neighborhood, :MSSubClass])

t = KoalaFlux.FrameToFluxInputTransformer()
tM = Machine(t, X)
XX = transform(tM, X[2:6,:])
@test XX.nominal ==  [2  3  4  1  5;
                      1  2  1  1  1]

for ftr in tM.scheme.nominal_features
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
train, test = split(eachindex(y), 0.7)
flux = FluxRegressor()
fluxM = Machine(flux, X, y, train)
fit!(fluxM, train)

flux.n = 30
flux.lambda = 1e-3
fit!(fluxM, train)
e_single = err(fluxM, train)

using KoalaEnsembles


