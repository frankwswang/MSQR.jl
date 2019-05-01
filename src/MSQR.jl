module MSQR
export MSQRpar
export MSQRtrain

# include("MPSSwapTest.jl")

using Yao, Yao.ConstGate
using MPSCircuit
push!(LOAD_PATH,abspath("src")) 
using MPSSwapTest
using Flux.Optimise

mutable struct MSQRpar
    circuit::ChainBlock
    reg::DefaultRegister
    vBit::Int64
    rBit::Int64
end

function MSQRtrain(par::MSQRpar, nMeasure::Int64, nTrain::Int64; learningRate=0.1, show::Bool=false)
    circuit = par.circuit
    regTar = par.reg
    vBit = par.vBit
    rBit = par.rBit
    dGates = collect_blocks(AbstractDiff, circuit)
    # println("t1")
    for i=1:nTrain
        mst = MSTest(regTar, circuit, vBit, rBit, nMeasure)
        # println("t2")
        grads = opdiff.(()->MSTest(regTar, circuit, vBit, rBit, nMeasure).regA, dGates,Ref(mst.witnessOp))
        # println("t3")
        if learningRate == "ADAM"
            # println("t3_1")
            dGatesPar = [parameters(dGates[i])[1] for i=1:length(dGates)]
            # println("t3_2")
            Optimise.update!(ADAM(), dGatesPar, (-1.0.*(dGatesPar.^2).*grads))
            # println("t3_3")
            dispatch!.(dGates, dGatesPar)
        else
            dispatch!.(+, dGates, grads.*learningRate)
        end
        # println("t4")
        showCase = (25*i)%nTrain==0
        if  showCase && show
            println("Training Step $(i), overlap = $(mst.overlaps)")
        end
        # println("t5")
    end
    circuit
end

end

