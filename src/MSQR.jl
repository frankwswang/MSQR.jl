module MSQR

using Yao, Yao.ConstGate
using MPSCircuit
using Flux.Optimise
using LinearAlgebra
using Statistics
using Test
using Requires

include("MPSSwapTest.jl")
include("MSQRtrain.jl")
include("SWAPtrain.jl")

@init @require CuYao="b48ca7a8-dd42-11e8-2b8e-1b7706800275" using CuYao
end