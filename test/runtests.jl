using Revise
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
flux.n = 100
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

u, v = @curve p linspace(0,0.5,11) begin
    flux.network_creator = get_creator(p)
    fit!(fluxM)
    err(fluxM, test, loss=rmsl)
end

p = u[indmin(v)] # 

flux.n=200
fit!(fluxM)
@test abs(err(fluxM, test, loss=rmsl) - 0.14) < 0.3



