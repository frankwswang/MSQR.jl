#=
Normal SWAP-Test algorithm related Functions that help verify the validity of MSQR training results.
=#
export SWAPtest, SWAPtrain!, SWAPtestRes


"""
    SWAPtestRes{overlap::Float64, witnessOp::PutBlock, reg::DefaultRegister, circuit::ChainBlock}
Fields:
\n`overlap::Float64`:     Overlap bewteen generated register and target register.
\n`witnessOp::PutBlock`:  The witness(measure) operator of the `SWAPtest` circuit.
\n`reg::DefaultRegister`: The register goes through `SWAPtest`. nqubits(SWAPtest.reg) = 1 + 2*nBitT.
\n`circuit::ChainBlock`:  Circuit of `SWAPtest`.
"""
struct SWAPtestRes
    overlap::Float64     # Overlap bewteen generated register and target register.
    witnessOp::PutBlock  # The witness(measure) operator of the `SWAPtest` circuit.
    reg::DefaultRegister # The register goes through `SWAPtest`. nqubits(SWAPtest.reg) = 1 + 2*nBitT.
    circuit::ChainBlock  # Circuit of `SWAPtest`.
end


"""
    SWAPtest(regG::DefaultRegister, regT::DefaultRegister; nMeasure::Int64=1, ϕ::Real=0, useCuYao::Bool=CUDA_ON)
    ->
    SWAPtestRes{overlap::Float64, witnessOp::PutBlock, reg::DefaultRegister, circuit::ChainBlock}
SWAP test function which get the overlap between target register(regT) and generated register(reg0).
\n`regG::DefaultRegister`: Generated quantum state to compare with target quantum state.
\n`regT::DefaultRegister`: Target quantum state.
"""
function SWAPtest(regG::DefaultRegister, regT::DefaultRegister; nMeasure::Int64=1, ϕ::Real=0, useCuYao::Bool=CUDA_ON)
    nMcheck(nMeasure, regG, regT)
    reg1 = repeat(copy(regG), nMeasure)
    reg2 = repeat(copy(regT), nMeasure)
    reg0 = zero_state(1, nbatch=nMeasure)
    regA = join(reg0, reg1, reg2)
    useCuYao && (regA = regA |> cu)
    resS = SWAPtest(regA, ϕ=ϕ)
    regA = resS.reg
    useCuYao && (regA = regA |> cpu)
    SWAPtestRes(resS.overlap, resS.witnessOp, regA, resS.circuit)
end
"""
    SWAPtest(regA::DefaultRegister; nMeasure::Int64=1, ϕ::Real=0)
    ->
    SWAPtestRes{overlap::Float64, witnessOp::PutBlock, reg::DefaultRegister, circuit::ChainBlock}
Method 2 of `SWAPtest`:
\n`regA::DefaultRegister`: Input register(state) for SWAP Test circuit. `regA = join(zero_state(1), reg1, reg2)`
"""
function SWAPtest(regA::DefaultRegister; nMeasure::Int64=1, ϕ::Real=0)
    regA = copy(repeat(regA, nMeasure))
    nBitA = nqubits(regA)
    nBitT = Int((nBitA - 1) / 2)
    witnessOp = put(nBitA, nBitA=>Z)
    SWAPblock = chain(nBitA, [control(nBitA, nBitA, (i,i-nBitT)=>ConstGate.SWAP) for i=nBitA-1:-1:nBitT+1])
    SWAPcircuit = chain(nBitA, put(nBitA, nBitA=>H), put(nBitA, nBitA=>shift(ϕ)), SWAPblock, put(nBitA, nBitA=>H))
    regA |> SWAPcircuit
    overlaps = expect(witnessOp, regA)
    overlap = mean(overlaps) |> real
    SWAPtestRes(overlap, witnessOp, regA, SWAPcircuit)
end
"""
    SWAPtest(regTar::DefaultRegister, circuit::ChainBlock; nMeasure::Int64=1, ϕ::Real=0, useCuYao::Bool=CUDA_ON)
    ->
    SWAPtestRes{overlap::Float64, witnessOp::PutBlock, reg::DefaultRegister, circuit::ChainBlock}
Method 3 of `SWAPtest`:
\n`regTar::DefaultRegister`: Target quantum state.
\n`circuit::ChainBlock`: circuit to generate the state(regG) to compare with `regTar`.
"""
function SWAPtest(regTar::DefaultRegister, circuit::ChainBlock; nMeasure::Int64=1, ϕ::Real=0, useCuYao::Bool=CUDA_ON)
    n = nqubits(regTar)
    regT = copy(regTar)
    regG = zero_state(nqubits(regT)) |> circuit
    SWAPtest(regG, regT, nMeasure=nMeasure, ϕ=ϕ, useCuYao=useCuYao)
end
function SWAPtest(circuit::ChainBlock; regAll::DefaultRegister, nMeasure::Int64=1, ϕ::Real=0)
    nMcheck(nMeasure, regAll)
    regA = copy(regAll)
    nBitT = Int((nqubits(regA) - 1) / 2)
    regA |> focus!(nBitT+1:2nBitT...) |> circuit |> relax!(nBitT+1:2nBitT...)
    SWAPtest(regA, nMeasure=nMeasure, ϕ=ϕ)
end


"""
    SWAPtrain!(regTar::DefaultRegister, circuit::ChainBlock, nTrain::Union{Int64, :auto}; nMeasure::Int64=1, Gmethod::String="Qdiff", GDmethod=("default",0.01), show::Bool=false, useCuYao::Bool=CUDA_ON) 
    -> 
    overlaps::Array{Float64,1}
SWAP-Test training function. This function will change the parameters of differentiable gates in circuit. When set `nTrain = :auto`, trigger the automaic training ieration. 
"""
function SWAPtrain!(regTar::DefaultRegister, circuit::ChainBlock, nTrain::Union{Int64, Symbol}; nMeasure::Int64=1,
                    Gmethod::Union{String, Tuple{String,Float64}}="Qdiff", GDmethod=("default",0.01), show::Bool=false, useCuYao::Bool=CUDA_ON)
    if show
        cPar = MPSDCpar(circuit)
        nBitT = cPar.nBitA
        vBit = cPar.vBit
        rBit = cPar.rBit
        depth = cPar.depth
        println("\nSWAP Training Parameters:")
        println("nBitT=$(nBitT) vBit=$(vBit) rBit=$(rBit) depth=$(depth)")
        println("nMeasure=$(nMeasure) nTrain=$(nTrain) GDmethod=$(GDmethod)\n")
        println("Initial overlap = $(SWAPtest(regTar, circuit, nMeasure=nMeasure, useCuYao=useCuYao).overlap)")
    end
    reg = zero_state(nqubits(regTar)+1, nbatch = nMeasure)
    regT = repeat(regTar, nMeasure)
    regA = copy(join(reg, regT))
    useCuYao == true && (regA = regA |> cu)
    if typeof(nTrain) == Int64
        res = train!(nTrain, circuit, Tmethod = circuit->SWAPtest(circuit, regAll = regA), 
                     Gmethod=Gmethod, GDmethod=GDmethod, show=show)
    elseif nTrain == :auto
        res = train!(nTrain, circuit, Tmethod = circuit->SWAPtest(circuit, regAll = regA), 
                     Gmethod=Gmethod, GDmethod=GDmethod, show=show)
    end
    res
end