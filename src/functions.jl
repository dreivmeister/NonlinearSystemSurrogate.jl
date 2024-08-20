using NonlinearSolve
using Flux
using ParameterSchedulers
using QuasiMonteCarlo
using LinearAlgebra
using Plots


function norm_loss(fup) # fup = f(u, p)
    return sum(norm.(eachcol(fup))) / size(fup, 2)
end

function create_surrogate(f, u_min, u_max, p_min, p_max; num_samples=1000, batchsize=25, numhiddenlayers=1, numhiddenunits=100, opt=Flux.Adam, lr=1e-4, epochs=100)
    # u*_hat = model(u0, p)
    u_dim = length(u_min)
    p_dim = length(p_min)
    up_min = vcat(u_min, p_min)
    up_max = vcat(u_max, p_max)

    # data
    up_data = Float32.(QuasiMonteCarlo.sample(num_samples, up_min, up_max, HaltonSample()))
    loader = Flux.DataLoader(up_data, batchsize=batchsize, shuffle=true)

    # model
    layers = [Dense(u_dim + p_dim, numhiddenunits, relu)]
    for i in 1:numhiddenlayers
        push!(layers, Dense(numhiddenunits, numhiddenunits, relu))
    end
    push!(layers, Dense(numhiddenunits, u_dim))
    model = Chain(layers...)

    # optim
    optim = Flux.setup(opt(lr), model)

    losses = []
    for epoch in 1:epochs
        for up in loader
            loss, grads = Flux.withgradient(model) do m
                u_hat = m(up)
                norm_loss(f(u_hat, up[u_dim+1:end,:]))
            end
            Flux.update!(optim, model, grads[1])
            push!(losses, loss)
        end
    end
    return model
end

