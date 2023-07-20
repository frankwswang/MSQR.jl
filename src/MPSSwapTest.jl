#=
Applying SWAP Test on a MPS-reusable circuit to get states' overlap between 2 wave functions.
=#
export MSCpar, MScircuit, MStestRes, MStest, MSTtestRes, MSTtest, nMcheck  


"""
    MSCpar(MSCircuit::ChainBlock)
    ->
    MSCpar{vBit::Int64, rBit::Int64, nBitT::Int64, nBitA::Int64, depth::Int64}
Get the set of nBit parameters from a circuit built by function `MScircuit`.
Fields:
\n`vBit::Int64`: Number of virtual qubits.
\n`rBit::Int64`: Number of reusable qubits.
\n`nBitT::Int64`: Number of qubits in target register.
\n`nBitA::Int64`: Number of qubits (lines) in MSCircuit.
"""
struct MSCpar
    vBit::Int64  # Number of virtual qubits.
    rBit::Int64  # Number of reusable qubits.
    nBitT::Int64 # Number of qubits in target register.
    nBitA::Int64 # Number of qubits (lines) in MSCircuit.
    depth::Int64 # Depth of each MPS block in MSCircuit.
    
    function MSCpar(MSCircuit::ChainBlock)
        nBitA = nqubits(MSCircuit)
        vBit = length(MSCircuit[end-1])
        rBit = length(MSCircuit[3][2])
        nBitT = nBitA - 1 - vBit - rBit
        if length(collect_blocks(QMPS.QDiff, MSCircuit)) == 0
            depth = 0
        else
            depth = length(content(MSCircuit[3][1])[1])
        end
        new(vBit, rBit, nBitT, nBitA, depth)
    end
end


"""
    MScircuit(nBitT::Int64, vBit::Int64, rBit::Int64, MPSblocks::Array{CompositeBlock,1}; ϕ::Float64=0.0) 
    -> 
    MSCircuit::ChainBlock
Function to build quantum circuit for MPS-Swap Test.
"""
function MScircuit(nBitT::Int64, vBit::Int64, rBit::Int64, MPSblocks::Array{CompositeBlock,1}; ϕ::Float64=0.0)
    par2nd = MPSpar(nBitT, vBit, rBit)
    nBlock = par2nd.nBlock
    nBitG = par2nd.nBit
    nBitA = 1 + nBitG + nBitT 
    Cblocks = AbstractBlock[]
    push!(Cblocks, put(nBitA, nBitA=>H))
    push!(Cblocks, put(nBitA, nBitA=>shift(ϕ)))
    for i = 1:nBlock
        # MPSblock = put(nBitA, Tuple( (nBitT+1):(nBitA-1), )=>MPSblocks[i] ) #Use `subroutine` instead of `put` for CuYao compatibility and efficiency.
        MPSblock = subroutine( nBitA, MPSblocks[i], (nBitT+1):(nBitA-1) )
        SWAPblock = chain(nBitA, [control(nBitA, nBitA, ( (nBitA-1-irBit), (nBitT-(i-1)*rBit-irBit) )=>SWAP) for irBit = 0:rBit-1])
        OpBlock = chain(nBitA, MPSblock, SWAPblock)
        push!(Cblocks, OpBlock)
        MeasureBlock = Measure(nBitA, locs=(nBitT+vBit+1):(nBitA-1), resetto=0)
        push!(Cblocks, MeasureBlock)
    end
    SWAPvBit = chain(nBitA, [control(nBitA, nBitA, ( (nBitT+i),i )=>SWAP) for i=vBit:-1:1])
    push!(Cblocks, SWAPvBit)
    push!(Cblocks, put(nBitA, nBitA=>H))
    MSCircuit = chain(nBitA,Cblocks)
    MSCircuit
end


"""
    MStestRes{reg::AbstractArrayReg, witnessOp::PutBlock, overlap::Float64}
Fields:
\n`reg::AbstractArrayReg`: The register goes through `MStest`. nqubits(MStest.reg) = 1 + vBit + rBit + nBitT.
\n`witnessOp::PutBlock`: The witness(measure) operator of the `MStest` circuit.
\n`overlap::Float64`: Overlap bewteen register generated by `MScircuit` and target register. 
"""
struct MStestRes
    reg::AbstractArrayReg # The register goes through `MStest`. nqubits(MStest.reg) = 1 + vBit + rBit + nBitT.
    witnessOp::PutBlock  # The witness(measure) operator of the `MStest` circuit.
    overlap::Float64     # Overlap bewteen register generated by MSCircuit and target register.
end


"""
    MStest(regT::AbstractArrayReg, MSCircuit::ChainBlock; nMeasure::Int64=1, useCuYao::Bool=CUDA_ON) 
    -> 
    MStestRes{reg::AbstractArrayReg, witnessOp::PutBlock, overlap::Float64}
Return the main structure of MPS-Swap Test algorithm.
\n `regT::AbstractArrayReg`: Target quantum state.
"""
function MStest(regT::AbstractArrayReg, MSCircuit::ChainBlock; nMeasure::Int64=1, useCuYao::Bool=CUDA_ON)
    nMcheck(nMeasure, regT)
    nBitA = MSCpar(MSCircuit).nBitA
    nBitT = nqubits(regT)  
    regA = join(zero_state(nBitA-nBitT, nbatch=nMeasure), clone(regT,nMeasure) ) #This operation changes regA in the parent scope.
    useCuYao && (regA = regA |> CuYao.cu)
    resS = MStest(MSCircuit, regAll=regA)
    witnessOp = resS.witnessOp
    ActualOverlap = resS.overlap
    regA = resS.reg
    useCuYao && (regA = regA |> CuYao.cpu)
    MStestRes(regA, witnessOp, ActualOverlap)
end
"""
    MStest(MSCircuit::ChainBlock; regAll::AbstractArrayReg) 
    -> 
    MStestRes{reg::AbstractArrayReg, witnessOp::PutBlock, overlap::Float64}
Method 2 of `MStest`:
\n `regAll::AbstractArrayReg`: Input register(state) for MSCircuit(MSQR circuit) built by `MScircuit`. `regAll = join(zero_state(1+nBitV+nBitR), regT)`
"""
function MStest(MSCircuit::ChainBlock; regAll::AbstractArrayReg)  
    regA = copy(regAll)
    nBitA = nqubits(regA)
    witnessOp = put(nBitA, nBitA=>Z)
    regA |> MSCircuit
    Overlaps = expect(witnessOp, regA)
    ActualOverlap = mean(Overlaps) |> real #Take the real part for simplicity since the imaginary part is 0.
    MStestRes(regA, witnessOp, ActualOverlap)
end


"""
    MSTtestRes{Aoverlap::Float64, Eoverlap::Float64, error::Float64}
Fields:
\n`Aoverlap::Float64`: Actual overlap generated by MPS-Swap Test.
\n`Eoverlap::Float64`: Expected overlap calculated through formula.
\n`error::Float64`:    Aoverlap/Eoverlap - 1
"""
struct MSTtestRes
    Aoverlap::Float64 # Actual overlap generated by MPS-Swap Test.
    Eoverlap::Float64 # Expected overlap calculated through formula.
    error::Float64    # Aoverlap/Eoverlap - 1
end


"""
    MSTtest(regT::AbstractArrayReg, MSCircuit::ChainBlock, cExtend::ChainBlock; nMeasure::Int64=1, useCuYao::Bool=CUDA_ON) 
    -> 
    MSTtestRes{Aoverlap::Float64, Eoverlap::Float64, error::Float64}
Test function that verify the validity of MPS-Swap Test algorithm.
"""
function MSTtest(regT::AbstractArrayReg, MSCircuit::ChainBlock, cExtend::ChainBlock; nMeasure::Int64=1, useCuYao::Bool=CUDA_ON)             
    MSTres = MStest(regT, MSCircuit, nMeasure=nMeasure, useCuYao=useCuYao)
    ActualOverlap = MSTres.overlap
    regG = zero_state(nqubits(regT))
    regG |> cExtend
    ExpectOverlap = ((regT.state'*regG.state)[1] |> abs)^2 #[1]: Convert array to complex number.
    error = ActualOverlap/ExpectOverlap - 1
    MSTtestRes(ActualOverlap, ExpectOverlap, error)
end


"""
    nMcheck(nMeasure::Int64, regs::AbstractArrayReg...)
Function that check if the user not only define nMeasure but also define nbatch.  
"""
function nMcheck(nMeasure::Int64, regs::AbstractArrayReg...)
    hasNoBatch = all(nbatch.(regs) .== Ref(NoBatch()))
    if nMeasure!=1 && !hasNoBatch
        print("WARNING: Measure Times has already been defined in nbatch(reg)!!!
              \nNow the actual Measure Times = nMeasure*nbatch(reg).\n")
    end
end


# Default setting of CuYao(CUDA) support is off.
CUDA_ON = false