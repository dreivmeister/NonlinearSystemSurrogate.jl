# NonlinearSystemSurrogate

[![Build Status](https://github.com/dreivmeister/NonlinearSystemSurrogate.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/dreivmeister/NonlinearSystemSurrogate.jl/actions/workflows/CI.yml?query=branch%3Amaster)



So you have a nonlinear system $f(u,p) = 0$ that you want to solve using NonlinearSolve.jl (dont think thats a must).
Maybe you need to solve it many times with different $p$ or $u$. Then you can use this package to train a neural network surrogate and use it to predict a root of $f(u,p) = 0$ given u and p like that: 
$surrogate(u0,p) = u_hat$.
Here $u0$ is an inital guess vector for the root and $u_hat$ is the predicted root vector $u$.
That can be used on its own or can be plugged into a solve routine like so: $NewtonSolve(f, u_hat, p) = u*$.

Doing this should reduce number of needed solving steps and therefore computation time.

Here a quick example with actual code:
