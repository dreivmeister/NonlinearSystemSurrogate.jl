# NonlinearSystemSurrogate

[![Build Status](https://github.com/dreivmeister/NonlinearSystemSurrogate.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/dreivmeister/NonlinearSystemSurrogate.jl/actions/workflows/CI.yml?query=branch%3Amaster)



So you have a nonlinear system $f(u,p) = 0$ that you want to solve using NonlinearSolve.jl (dont think thats a must).
Maybe you need to solve it many times with different $p$ or $u$. Then you can use this package to train a neural network surrogate and use it to predict a root of $f(u,p) = 0$ given u and p like that: 
$surrogate(u0,p) = u_{pred}$.
Here $u0$ is an inital guess vector for the root and $u_{pred}$ is the predicted root vector $u$.
That can be used on its own or can be plugged into a solve routine like so: $NewtonSolve(f, u_{pred}, p) = u$.

Doing this should reduce number of needed solving steps and therefore computation time.

Here a quick example with actual code:

First we setup our function $f$ of which we want to find the root together with an initial guess vector $u0$ and a parameter vector $p$.
```
using NonlinearSolve

function f(u, p)
    u .^ 3 .- p
end
u0 = rand(2)
p = rand(2)
```
Then we find a root using NonlinearSolve.jl and then look at how many iterations it took.
```
prob = NonlinearProblem(f, u0, p)
sol = solve(prob, NewtonRaphson())

println(sol.stats.nsteps) # 4
```
Now we can train a surrogate by passing $f$ and min and max bounds for $u$ and $p$ into $create_surrogate$.
```
up_min = [-2.,-2.,-2.,-2.] # the min bounds for u and p (first two numbers are for u, second two for p)
up_max = [2.,2.,2.,2.] # the max bounds for u and p
model = train_surrogate(f, up_min, up_max)
```
When that is done, we can solve it again using the models prediction as the new $u0$ and hope that the number of iterations until convergence decreased.
```
prob = NonlinearProblem(f, model(vcat(u0,p)), p)
sol = solve(prob, NewtonRaphson())

println(sol.stats.nsteps) # should be <4
```
