#=
Combining MPS-SwapTest method and Quantum Gradient Optimization, 
train a MPS circuit with adjustable parameters to recreate target wave function.
=#
export MSQRtrain!, GDescent


"""
    MSQRtrain!(regTar::ArrayReg, MSCircuit::ChainBlock, nTrain::Union{Int64, :auto}; nMeasure::Int64=1, Gmethod::String="Qdiff", GDmethod=("default",0.01), show::Bool=false, useCuYao::Bool=CUDA_ON, ConvTh::Tuple{Float64, Float64}=(5e-4, 1e-3)) 
    -> 
    overlaps::Array{Float64,1}
MSQR training function. This function will change the parameters of differentiable gates in MSCircuit. When set `nTrain = :auto`, trigger the automaic training ieration.
"""
function MSQRtrain!(regTar::ArrayReg, MSCircuit::ChainBlock, nTrain::Union{Int64, Symbol}; nMeasure::Int64=1,
                    Gmethod::Union{String, Tuple{String,Float64}}="Qdiff", GDmethod=("default",0.01), show::Bool=false, 
                    useCuYao::Bool=CUDA_ON, ConvTh::Tuple{Float64, Float64}=(5e-4, 1e-3))
    cPar = MSCpar(MSCircuit)
    vBit = cPar.vBit
    rBit = cPar.rBit
    if show
        nBitT = cPar.nBitT
        depth = cPar.depth
        println("\nMSQR Training Parameters:")
        println("nBitT=$(nBitT) vBit=$(vBit) rBit=$(rBit) depth=$(depth)")
        println("nMeasure=$(nMeasure) nTrain=$(nTrain) GDmethod=$(GDmethod)\n")
        println("Initial overlap = $(MStest(regTar, MSCircuit, nMeasure=nMeasure, useCuYao=useCuYao).overlap)")
    end
    reg0 = zero_state((vBit+rBit+1), nbatch=nMeasure)
    regT = repeat(copy(regTar), nMeasure)
    regA = join(reg0, regT)
    useCuYao == true && (regA = regA |> cu)
    if typeof(nTrain) == Int64
        res = train!(nTrain, MSCircuit, Tmethod = MSCircuit->MStest(MSCircuit, regAll=regA), 
                     Gmethod=Gmethod, GDmethod=GDmethod, show=show)
    elseif nTrain == :auto
        res = train!(MSCircuit, Tmethod = MSCircuit->MStest(MSCircuit, regAll=regA), 
                     Gmethod=Gmethod, GDmethod=GDmethod, show=show, ConvTh=ConvTh)
    end
    res
end


"""
    getDiffs!(circuit::ChainBlock, dGates::Array{QMPS.QDiff, 1}, witnessOp::Yao.Add{<:Any}; trainf::function, showStep::Bool=false, Gmethod::Union{String, Tuple{String,Float64}}="Qdiff", j::Int64=1)
    ->
    tm.overlaps::Float64
Core function of train! to get the overlap in each iteration.
"""
@inline function getDiffs!(circuit::ChainBlock, dGates::Array{QMPS.QDiff, 1}, witnessOp::Yao.Add{<:Any}, 
                   trainf::Function; GDf::Function=GDescent(("default",0.01)),
                   showStep::Bool=false, Gmethod::Union{String, Tuple{String,Float64}}="Qdiff", j::Int64=1)
    if Gmethod == "Qdiff"
        grads = getQdiff!.(()->trainf(circuit).reg, dGates, Ref(witnessOp))
    elseif Gmethod == "Ndiff"
        grads = getNdiff!.(()->trainf(circuit).reg, dGates, Ref(witnessOp))
    elseif Gmethod[1] == "Ndiff"
        grads = getNdiff!.(()->trainf(circuit).reg, dGates, Ref(witnessOp), δ=Gmethod[2])
    end 
    dGatesPar = [parameters(dGates[j])[1] for j=1:length(dGates)]
    dGatesPar = GDf(dGatesPar, grads)
    dispatch!.(dGates, dGatesPar)
    tm = trainf(circuit)
    if  showStep
        println("Training Step $(j), overlap = $(tm.overlap)")
    end
    tm.overlap
end


"""
    train!(nTrain::Int64, circuit::ChainBlock; Tmethod::Function, Gmethod::Union{String, Tuple{String,Float64}}="Qdiff", GDmethod=("default",0.01), show::Bool=false) 
    -> 
    overlaps::Array{Float64,1}
Function that trains parameters of differentiable gates in the input circuit with designated overlap method 
in order to make the circuit generate a similar wave function to the target wave function. 
"""
function train!(nTrain::Int64, circuit::ChainBlock; Tmethod::Function, 
                Gmethod::Union{String, Tuple{String,Float64}}="Qdiff", GDmethod=("default",0.01), show::Bool=false)
    overlaps = Float64[]
    GD = GDescent(GDmethod)
    tm=Tmethod(circuit)
    witnessOp = put(nqubits(tm.witnessOp), 1=>I2) - tm.witnessOp
    dGates = collect_blocks(QMPS.QDiff, circuit)
    for i=1:nTrain
        # if Gmethod == "Qdiff"
        #     grads = getQdiff!.(()->Tmethod(circuit).reg, dGates, Ref(witnessOp))
        # elseif Gmethod == "Ndiff"
        #     grads = getNdiff!.(()->Tmethod(circuit).reg, dGates, Ref(witnessOp))
        # elseif Gmethod[1] == "Ndiff"
        #     grads = getNdiff!.(()->Tmethod(circuit).reg, dGates, Ref(witnessOp), δ=Gmethod[2])
        # end 
        # dGatesPar = [parameters(dGates[j])[1] for j=1:length(dGates)]
        # dGatesPar = GD(dGatesPar, grads) 
        # dispatch!.(dGates, dGatesPar)
        # tm = Tmethod(circuit)
        # showCase = (25*i)%nTrain==0
        # if  showCase && show
        #     println("Training Step $(i), overlap = $(tm.overlap)")
        # end
        showCase = (25*i)%nTrain==0
        overlap = getDiffs!(circuit, dGates, witnessOp, Tmethod, GDf=GD, showStep=(showCase && show), Gmethod=Gmethod, j=i)
        push!(overlaps,overlap)
    end
    overlaps
end
"""
    train!(circuit::ChainBlock; Tmethod::Function, Gmethod::Union{String, Tuple{String,Float64}}="Qdiff", GDmethod=("default",0.01), show::Bool=false, ConvTh::Tuple{Float64, Float64}=(5e-4, 1e-3)) 
    -> 
    overlaps::Array{Float64,1}
Method 2 of `train!`:
\nAutomatically stop training when objective function reaches maximum value. 
"""
function train!(circuit::ChainBlock; Tmethod::Function, 
                Gmethod::Union{String, Tuple{String,Float64}}="Qdiff", GDmethod=("default",0.01), show::Bool=false, ConvTh::Tuple{Float64, Float64}=(5e-4, 1e-3))
    overlaps = Float64[]
    GD = GDescent(GDmethod)
    tm=Tmethod(circuit)
    witnessOp = put(nqubits(tm.witnessOp), 1=>I2) - tm.witnessOp
    dGates = collect_blocks(QMPS.QDiff, circuit)
    nTrain = 200
    gap = 199
    @label train
    for i=nTrain-gap:nTrain
        # if Gmethod == "Qdiff"
        #     grads = getQdiff!.(()->Tmethod(circuit).reg, dGates, Ref(witnessOp))
        # elseif Gmethod == "Ndiff"
        #     grads = getNdiff!.(()->Tmethod(circuit).reg, dGates, Ref(witnessOp))
        # elseif Gmethod[1] == "Ndiff"
        #     grads = getNdiff!.(()->Tmethod(circuit).reg, dGates, Ref(witnessOp), δ=Gmethod[2])
        # end 
        # dGatesPar = [parameters(dGates[j])[1] for j=1:length(dGates)]
        # dGatesPar = GD(dGatesPar, grads) 
        # dispatch!.(dGates, dGatesPar)
        # tm = Tmethod(circuit)
        # showCase = (25*i)%nTrain==0
        # if  showCase && show
        #     println("Training Step $(i), overlap = $(tm.overlap)")
        # end
        # push!(overlaps,tm.overlap)
        showCase = (25*i)%nTrain==0
        overlap = getDiffs!(circuit, dGates, witnessOp, Tmethod, GDf=GD, showStep=(showCase && show), Gmethod=Gmethod, j=i)
        push!(overlaps,overlap)
    end
    gap = 99
    if abs(mean(overlaps[nTrain-199:nTrain-100]) - mean(overlaps[nTrain-99:nTrain])) > ConvTh[1] || 
       abs(mean(overlaps[nTrain-199:nTrain-150]) - mean(overlaps[nTrain-49:nTrain])) > ConvTh[1]*2 ||
       overlaps[nTrain-49:nTrain] |> std |> abs > ConvTh[2]
        nTrain = nTrain+100
        @goto train
    end
    overlaps
end


"""
    GDescent(GDmethod::Union{String, Tuple{String,Float64}, Tuple{String,Float64,Tuple{Float64,Float64}}}) 
    -> 
    reF::Function
Function that returns the aimed Gradient Descent method.
"""
function GDescent(GDmethod::Union{String, Tuple{String,Float64}, Tuple{String,Float64,Tuple{Float64,Float64}}})
    if GDmethod == "ADAM"
        resF = (dGatesPar, grads) -> Flux.Optimise.update!(ADAM(), dGatesPar, grads)
    elseif GDmethod[1] == "ADAM"
        if length(GDmethod) == 2
            resF = (dGatesPar, grads) -> Flux.Optimise.update!(ADAM(GDmethod[2]), dGatesPar, grads)
        elseif length(GDmethod) == 3    
            resF = (dGatesPar, grads) -> Flux.Optimise.update!(ADAM(GDmethod[2],GDmethod[3]), dGatesPar, grads)
        end    
    elseif GDmethod[1] == "default" && length(GDmethod) == 2
        resF = (dGatesPar, grads) -> dGatesPar - grads.*GDmethod[2]
    end
    resF
end