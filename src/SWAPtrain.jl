#=
Normal SWAP-Test algorithm related Functions that help verify the validity of MSQR training results.
=#
export SWAPtest, SWAPtrain!


"""
    SWAPtest(reg0::DefaultRegister, regT::DefaultRegister, nMeasure::Int64; ϕ::Real=0)
SWAP test function which get the overlap between target register(regT) and generated register(reg0).
\nFields:
\n`overlap::Float64`:     Overlap bewteen generated register and target register.
\n`witnessOp::PutBlock`:  The witness(measure) operator of the `SWAPtest` circuit.
\n`reg::DefaultRegister`: The register goes through `SWAPtest`. nqubits(SWAPtest.reg) = 1 + 2*nBitT.
\n`circuit::ChainBlock`:  Circuit of `SWAPtest`.
"""
struct SWAPtest
    overlap::Float64     # Overlap bewteen generated register and target register.
    witnessOp::PutBlock  # The witness(measure) operator of the `SWAPtest` circuit.
    reg::DefaultRegister # The register goes through `SWAPtest`. nqubits(SWAPtest.reg) = 1 + 2*nBitT.
    circuit::ChainBlock  # Circuit of `SWAPtest`.

    function SWAPtest(reg0::DefaultRegister, regT::DefaultRegister, nMeasure::Int64; ϕ::Real=0)
        reg1 = repeat(copy(reg0), nMeasure)
        reg2 = repeat(copy(regT), nMeasure)
        nBitT = nqubits(reg2)
        nBitA = nBitT*2 + 1
        witnessOp = put(nBitA, nBitA=>Z)
        SWAPblock = chain(nBitA, [control(nBitA, nBitA, (i,i-nBitT)=>ConstGate.SWAP) for i=nBitA-1:-1:nBitT+1])
        cSWAP = chain(nBitA, put(nBitA, nBitA=>H), put(nBitA, nBitA=>shift(ϕ)), SWAPblock, put(nBitA, nBitA=>H))
        reg = join(repeat(zero_state(1),nMeasure), reg1)
        reg = join(reg, reg2)
        reg |> cSWAP
        overlapOs = expect(witnessOp, reg)
        overlap = sqrt(mean(overlapOs) |> real)
        new(overlap, witnessOp, reg, cSWAP)
    end

    function SWAPtest(regT::DefaultRegister, circuit::ChainBlock, nMeasure::Int64; ϕ::Real=0)
        reg0 = zero_state(nqubits(regT))
        reg1 = repeat(copy(reg0 |> circuit), nMeasure)
        reg2 = repeat(copy(regT), nMeasure)
        nBitT = nqubits(reg2)
        nBitA = nBitT*2 + 1
        witnessOp = put(nBitA, nBitA=>Z)
        SWAPblock = chain(nBitA, [control(nBitA, nBitA, (i,i-nBitT)=>ConstGate.SWAP) for i=nBitA-1:-1:nBitT+1])
        cSWAP = chain(nBitA, put(nBitA, nBitA=>H), put(nBitA, nBitA=>shift(ϕ)), SWAPblock, put(nBitA, nBitA=>H))
        reg = join(repeat(zero_state(1),nMeasure), reg1)
        reg = join(reg, reg2)
        reg |> cSWAP
        overlapOs = expect(witnessOp, reg)
        overlap = sqrt(mean(overlapOs) |> real)
        new(overlap, witnessOp, reg, cSWAP)
    end
end


"""
    SWAPtrain!(regTar::DefaultRegister, circuit::ChainBlock, nMeasure::Int64, nTrain::Int64; Gmethod::String="Qdiff", GDmethod=("default",0.01), show::Bool=false) -> overlaps::Array{Float64,1}
SWAP-Test training function. This function will change the parameters of differentiable gates in circuit.
"""
function SWAPtrain!(regTar::DefaultRegister, circuit::ChainBlock, nMeasure::Int64, nTrain::Int64;
                    Gmethod::String="Qdiff", GDmethod=("default",0.01), show::Bool=false)
    if show
        cPar = MPSDCpar(circuit)
        nBitT = cPar.nBitA
        vBit = cPar.vBit
        rBit = cPar.rBit
        depth = cPar.depth 
        println("\nSWAP Training Parameters:")
        println("nBitT=$(nBitT) vBit=$(vBit) rBit=$(rBit) depth=$(depth)")
        println("nMeasure=$(nMeasure) nTrain=$(nTrain) GDmethod=$(GDmethod)\n")
        println("Initial overlap = $(SWAPtest(regTar, circuit, nMeasure).overlap)")
    end
    res = train!(nTrain, circuit, Tmethod = circuit->SWAPtest(regTar, circuit, nMeasure), 
                 Gmethod=Gmethod, GDmethod=GDmethod, show=show)
end