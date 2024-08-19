using NonlinearSolve
using Flux
using ParameterSchedulers
using QuasiMonteCarlo
using LinearAlgebra
using Plots


function norm_loss(fup) # fup = f(u, p)
    return sum(norm.(eachcol(fup))) / size(fup, 2)
end

function create_surrogate(f, u_min, u_max, p_min, p_max)
    # u*_hat = model(u0, p)
    return model
end

