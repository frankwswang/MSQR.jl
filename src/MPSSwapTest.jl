"""
Applying Swap Test on a MPS-reusable circuit to get state overlaps between 2 wave functions.

"""

export MScircuit
export MSTest
export MSTTest

# Function to build quantum circuit for MPS-Swap Test.
function MScircuit(nBitT::Int64, vBit::Int64, rBit::Int64, ϕ::Real, MPSblocks::Array)
    par2nd = setMPSpar(nBitT, vBit, rBit)
    nBlock = par2nd.nBlock
    nBitG = par2nd.nBit
    nBitA = 1 + nBitG + nBitT 
    Cblocks = []
    push!(Cblocks, put(nBitA, nBitA=>H))
    push!(Cblocks, put(nBitA, nBitA=>shift(ϕ)))
    # println("S1")
    for i = 1:nBlock
        MPSblock = put(nBitA, Tuple( (nBitT+1):(nBitA-1), )=>MPSblocks[i] )
        # println("S2")
        push!(Cblocks, MPSblock)
        SWAPblock = chain(nBitA, [control(nBitA, nBitA, ( (nBitA-1-irBit), (nBitT-(i-1)*rBit-irBit) )=>ConstGate.SWAP) for irBit = 0:rBit-1])
        push!(Cblocks, SWAPblock)
        # println("S3")
    end
    # println("S4")
    SWAPvBit = chain(nBitA, [control(nBitA, nBitA, ( (nBitT+i),i )=>ConstGate.SWAP) for i=vBit:-1:1])
    # println("S5")
    push!(Cblocks, SWAPvBit)
    push!(Cblocks, put(nBitA, nBitA=>H))
    # println("S6")
    circuit = chain(nBitA,Cblocks)
    # println("The circuit:\n $(circuit)")
    circuit
end

# Main function of MPS-Swap Test algorithm.
struct MSTest
    regA
    witnessOp
    overlaps

    function MSTest(regT::DefaultRegister, circuit::ChainBlock, vBit::Int64, rBit::Int64, nMeasure::Int64)
        nBitT = nqubits(regT)
        par2nd = setMPSpar(nBitT, vBit, rBit)
        nBlock = par2nd.nBlock
        nBitA = 1 + rBit + vBit + nBitT
        regA = zero_state(nBitA)
        witnessOp = put(nBitA, nBitA=>Z)
        # println("S7") 
        regA = join(zero_state(1+rBit+vBit, nbatch=nMeasure), repeat(copy(regT),nMeasure) ) #This operation changes regA in the parent scope.
        regA |> circuit[1] |> circuit[2]
        # println("S8")
        for i = 3:2:( 3 + 2*(nBlock-2) )
            regA |> circuit[i] |> circuit[i+1]
            measure_collapseto!(regA, Tuple(nBitA-rBit:nBitA-1,), config = 0) 
            # println("S9")
        end    
        for i = ( 3 + 2*(nBlock-1) ):length(circuit)
            regA |> circuit[i]
            # println("S10")
        end
        Overlaps = expect(witnessOp, regA)
        # println("S11")
        ActualOverlaps = mean(Overlaps) |> real #Take the real part for simplicity since the imaginary part is 0.
        new(regA, witnessOp, ActualOverlaps)
    end

end

# Test function that verify the validity of MPS-Swap Test algorithm. 
function MSTTest(regT::DefaultRegister, circuit::ChainBlock, cExtend::ChainBlock, 
                 vBit::Int64, rBit::Int64, nMeasure::Int64)             
    MSTres = MSTest(regT, circuit, vBit, rBit, nMeasure)
    println("\nThe circuit of MPSSwapTest:\n$(circuit)")
    ActualOverlaps = MSTres.overlaps
    nBitT = nqubits(regT)
    @testset "MPS-Swap Test Reliability Check" begin
        regG = zero_state(nBitT)
        regG |> cExtend
        ExpectOverlaps = ((regT.state'*regG.state)[1] |> abs)^2
        @test isapprox.(ActualOverlaps, ExpectOverlaps, atol=0.02*ExpectOverlaps)
        println("\nExpectOverlaps: $(ExpectOverlaps)\nActualOverlaps: $(ActualOverlaps)\n")
        res = ActualOverlaps 
    end
end