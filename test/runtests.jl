push!(LOAD_PATH, abspath("./src"))
push!(LOAD_PATH, abspath("../QMPS/src"))

using Test, QMPS, Yao, Random, Statistics
using MSQR

# Un-comment if testing GPU acceleration compatibility.
# using CuYao

@testset "SWAPtest(SWAPtrain.jl)" begin
    seedNum = 1234
    m = 1
    n = 3
    Random.seed!(seedNum)
    cgen = dispatch!(DCbuilder(n,4).circuit, :random) 
    reg1 = rand_state(n)
    reg2 = rand_state(n)
    reg3 = zero_state(n) |> cgen  
    olp12m1 = SWAPtest(reg1, reg2, nMeasure=m)
    olp12m2 = SWAPtest(join(zero_state(1), reg1, reg2), nMeasure=m)
    olp23m1 = SWAPtest(reg2, cgen, nMeasure=m)
    olp23m2 = SWAPtest(cgen, regAll = join(zero_state(n+1), reg2), nMeasure=m)
    
    cST = chain(7, put(7, 7=>H),
                   put(7, 7=>shift(0)), 
                   chain(7, control(7, 7, (6,3)=>SWAP), 
                            control(7, 7, (5,2)=>SWAP), 
                            control(7, 7, (4,1)=>SWAP)), 
                   put(7, 7=>H))
    op = put(2n+1, 2n+1=>Z)
    reg12t = repeat(join(zero_state(1), reg1, reg2), m) |> cST
    reg23t = repeat(join(zero_state(1), reg3, reg2), m) |> cST
    olp12t = expect(op, reg12t) |> mean |> real
    olp23t = expect(op, reg23t) |> mean |> real

    @test olp12m1.reg ≈ olp12m2.reg
    @test olp23m1.reg ≈ olp23m2.reg
    @test olp12m1.reg ≈ reg12t
    @test olp23m1.reg ≈ reg23t 
    @test olp12m2.witnessOp == op
    @test olp12m2.circuit == cST
    @test isapprox(olp12m1.overlap, olp12m2.overlap, atol=10e-12)
    @test isapprox(olp23m1.overlap, olp23m2.overlap, atol=10e-12)
    @test isapprox(olp12m1.overlap, olp12t, atol=10e-12)
    @test isapprox(olp23m1.overlap, olp23t, atol=10e-12)
end

@testset "MPSSwapTest.jl" begin
    seedNum = 1234
    # MScircuit(nBitT::Int64, vBit::Int64, rBit::Int64, MPSblocks::Array{CompositeBlock,1}; ϕ::Float64=0.0)
    ## MPSblcoks = MPSC("CS", 4, 1, 1).mpsBlocks 
    n0 = 4
    v0 = 1
    r0 = 1
    Random.seed!(seedNum)
    mps0cs = MPSC("CS", n0, v0, r0)
    csMblocks = mps0cs.mpsBlocks
    CSc = MScircuit(n0, v0, r0, mps0cs.mpsBlocks)
    CSct = chain(7, put(7, 7=>H),
                   chain(7, subroutine(7, csMblocks[1], 5:6), control(7, 7, (4,6)=>SWAP), Measure(7, locs=(6), resetto=0)),
                   chain(7, subroutine(7, csMblocks[2], 5:6), control(7, 7, (3,6)=>SWAP), Measure(7, locs=(6), resetto=0)), 
                   chain(7, subroutine(7, csMblocks[3], 5:6), control(7, 7, (2,6)=>SWAP), Measure(7, locs=(6), resetto=0)),
                   control(7, 7, (1,5)=>SWAP),
                   put(7, 7=>H))           
    opx(n) = chain(n, [put(n, i=>X) for i=1:n])
    opy(n) = chain(n, [put(n, i=>Y) for i=1:n])
    opz(n) = chain(n, [put(n, i=>Z) for i=1:n])
    function cr_test(reg::ArrayReg, cr::CompositeBlock, crt::CompositeBlock)
        n = nqubits(reg)
        Random.seed!(seedNum)
        reg1x = expect(opx(n), copy(reg) |> cr) |> mean |> real
        Random.seed!(seedNum)
        reg1y = expect(opy(n), copy(reg) |> cr) |> mean |> real
        Random.seed!(seedNum)
        reg1z = expect(opz(n), copy(reg) |> cr) |> mean |> real
        Random.seed!(seedNum)
        reg2x = expect(opx(n), copy(reg) |> crt) |> mean |> real
        Random.seed!(seedNum)
        reg2y = expect(opy(n), copy(reg) |> crt) |> mean |> real
        Random.seed!(seedNum)
        reg2z = expect(opz(n), copy(reg) |> crt) |> mean |> real
        @test reg1x ≈ reg2x
        @test reg1y ≈ reg2y
        @test reg1z ≈ reg2z
    end
    cr_test(repeat(join(zero_state(3),rand_state(4)),5000), CSc, CSct)
    
    ## MPSblcoks = MPSC(("DC",2), 8, 2, 2).mpsBlocks
    n1 = 8
    v1 = 2
    r1 = 2
    d1 = 2
    Random.seed!(seedNum)
    mps0dc = MPSC(("DC",d1), n1, v1, r1)
    dcMblocks = mps0dc.mpsBlocks
    DCc = MScircuit(n1, v1, r1, mps0dc.mpsBlocks)
    DCct = chain(13, put(13, 13=>H),
                     chain(13, subroutine(13, dcMblocks[1], 9:12), control(13, 13, (12, 8)=>SWAP), control(13, 13, (11, 7)=>SWAP), Measure(13, locs=(11,12), resetto=0)),
                     chain(13, subroutine(13, dcMblocks[2], 9:12), control(13, 13, (12, 6)=>SWAP), control(13, 13, (11, 5)=>SWAP), Measure(13, locs=(11,12), resetto=0)),
                     chain(13, subroutine(13, dcMblocks[3], 9:12), control(13, 13, (12, 4)=>SWAP), control(13, 13, (11, 3)=>SWAP), Measure(13, locs=(11,12), resetto=0)),
                     chain(13, control(13, 13, (10, 2)=>SWAP), control(13, 13, (9, 1)=>SWAP)),
                     put(13, 13=>H))
    cr_test(repeat(join(zero_state(5),rand_state(8)),5000), DCc, DCct)
    
    # MStest(regT::ArrayReg, MSCircuit::ChainBlock; nMeasure::Int64=1, useCuYao::Bool=CUDA_ON)
    # MStest(MSCircuit::ChainBlock; regAll::ArrayReg)
    # MSTtest(regT::ArrayReg, MSCircuit::ChainBlock, cExtend::ChainBlock; nMeasure::Int64=1, useCuYao::Bool=CUDA_ON)
    ## MPSblcoks = MPSC("CS", 4, 1, 1).mpsBlocks
    m = 500
    n = 4
    v = 1
    r = 1
    Random.seed!(seedNum)
    regT = rand_state(n)
    mps1 = MPSC("CS", n, v, r)
    c1 = MScircuit(n, v, r, mps1.mpsBlocks)
    Random.seed!(seedNum)
    t1_0_m1 = MStest(regT, c1, nMeasure=m)
    Random.seed!(seedNum)
    t1_0_m2 = MStest(c1, regAll=repeat(join(zero_state(nqubits(c1)-n),regT),m))
    Random.seed!(seedNum)
    t1 = MSTtest(regT, c1, mps1.cExtend, nMeasure=m)
    @test t1_0_m1.reg ≈ t1_0_m2.reg
    @test t1_0_m1.witnessOp == t1_0_m2.witnessOp
    @test isapprox(t1_0_m1.overlap, t1_0_m2.overlap, atol=10e-12)
    @test isapprox(t1_0_m1.overlap, t1.Aoverlap, atol=10e-12)
    @test isapprox(t1.Eoverlap, t1.Aoverlap, atol=1.001abs(t1.error*t1.Eoverlap))
    Ext_overlap = SWAPtest(zero_state(n)|>mps1.cExtend, regT, nMeasure=m).overlap
    @test isapprox.(t1.Eoverlap, Ext_overlap, atol=1.001abs(t1.error*t1.Eoverlap))
    
    ## MPSblcoks = MPSC(("DC",2), 6, 2, 2).mpsBlocks
    n = 4
    v = 2
    r = 1
    d = 3
    Random.seed!(seedNum)
    regT = rand_state(n)
    mps2 = MPSC(("DC",d), n, v, r)
    c2 = MScircuit(n, v, r, mps2.mpsBlocks)
    p = MSCpar(c2)
    @test (p.nBitA, p.nBitT, p.vBit, p.rBit, p.depth) == (nqubits(c2), n, v, r, d)
    Random.seed!(seedNum)
    t2_0_m1 = MStest(regT, c2, nMeasure=m)
    Random.seed!(seedNum)
    t2_0_m2 = MStest(c2, regAll=repeat(join(zero_state(p.nBitA-n),regT),m))
    Random.seed!(seedNum)
    t2 = MSTtest(regT, c2, mps2.cExtend, nMeasure=m)
    @test t2_0_m1.reg ≈ t2_0_m2.reg
    @test t2_0_m1.witnessOp == t2_0_m2.witnessOp
    @test isapprox(t2_0_m1.overlap, t2_0_m2.overlap, atol=10e-12)
    @test isapprox(t2_0_m1.overlap, t2.Aoverlap, atol=10e-12)
    @test isapprox(t2.Eoverlap, t2.Aoverlap, atol=1.001abs(t2.error*t2.Eoverlap))
    Ext_overlap = SWAPtest(zero_state(n)|>mps2.cExtend, regT, nMeasure=m).overlap
    @test isapprox.(t2.Eoverlap, Ext_overlap, atol=1.001abs(t2.error*t2.Eoverlap))
end

@testset "MSQRtrain!(MSQRtrain.jl) + SWAPtrain!(SWAPtrain.jl)" begin
    seedNum = 1234
    n = 3
    v = 1
    r = 1
    d = 2
    Random.seed!(seedNum)
    regT = rand_state(n)
    Random.seed!(seedNum)
    mps = MPSC(("DC",d),n,v,r)
    c = MScircuit(n, v, r, mps.mpsBlocks)
    md1 = ("ADAM", 0.05)
    md2 = ("default", 0.2)
    m = 150

    mps_m1q = deepcopy(mps)
    c_m1q = MScircuit(n, v, r, mps_m1q.mpsBlocks)
    mps_s1q = deepcopy(mps)
    c_s1q = mps_s1q.cExtend
    mps_m2q = deepcopy(mps)
    c_m2q = MScircuit(n, v, r, mps_m2q.mpsBlocks)
    mps_s2q = deepcopy(mps)
    c_s2q = mps_s2q.cExtend
    mps_m1n = deepcopy(mps)
    c_m1n = MScircuit(n, v, r, mps_m1n.mpsBlocks)
    mps_s1n = deepcopy(mps)
    c_s1n = mps_s1n.cExtend
    mps_m2n = deepcopy(mps)
    c_m2n = MScircuit(n, v, r, mps_m2n.mpsBlocks)
    mps_s2n = deepcopy(mps)
    c_s2n = mps_s2n.cExtend

    showON = false
    Random.seed!(seedNum)
    mres1q = MSQRtrain!(regT, c_m1q, 40, nMeasure=m, GDmethod = md1, show=true) # Test show option.
    Random.seed!(seedNum)
    mres2q = MSQRtrain!(regT, c_m2q, 50, nMeasure=m, GDmethod = md2, show=showON)
    Random.seed!(seedNum)
    sres1q = SWAPtrain!(regT, c_s1q, 40, nMeasure=m, GDmethod = md1, show=true) # Test show option.
    Random.seed!(seedNum)
    sres2q = SWAPtrain!(regT, c_s2q, 50, nMeasure=m, GDmethod = md2, show=showON)
    Random.seed!(seedNum)
    mres1n = MSQRtrain!(regT, c_m1n, 40, nMeasure=m, GDmethod = md1, Gmethod=("Ndiff", 0.05), show=showON)
    Random.seed!(seedNum)
    mres2n = MSQRtrain!(regT, c_m2n, 50, nMeasure=m, GDmethod = md2, Gmethod=("Ndiff", 0.05), show=showON)
    Random.seed!(seedNum)
    sres1n = SWAPtrain!(regT, c_s1n, 40, nMeasure=m, GDmethod = md1, Gmethod=("Ndiff", 0.05), show=showON)
    Random.seed!(seedNum)
    sres2n = SWAPtrain!(regT, c_s2n, 50, nMeasure=m, GDmethod = md2, Gmethod=("Ndiff", 0.05), show=showON)
    ## Test different methods / optional arguments of the functions.
    c_m1n_2 = MScircuit(n, v, r, deepcopy(mps).mpsBlocks)
    Random.seed!(seedNum)
    a = MSQRtrain!(regT, c_m1n_2, 150, nMeasure=200, GDmethod = ("ADAM", 0.05, (0.9, 0.999)), Gmethod="Ndiff", show=showON)
    # @show a[end-4:end]
    mres1n_2 = MSQRtrain!(regT, c_m1n_2, :auto, nMeasure=m, GDmethod = md1, show=showON, ConvTh=(5e-3,1e-2))
    # @show length(mres1n_2)
    c_s1n_2 = deepcopy(mps).cExtend
    Random.seed!(seedNum)
    b = SWAPtrain!(regT, c_s1n_2, 20, nMeasure=m, GDmethod = md1, show=showON)
    # @show b[end-4:end]
    sres1n_2 = SWAPtrain!(regT, c_s1n_2, :auto, nMeasure=m, GDmethod = "ADAM", show=showON, ConvTh=(5e-3,1e-2))
    # @show length(sres1n_2)

    # If all the trainings converge in the end.
    cuti = 9
    tol = 0.06
    mres1qConv = mres1q[end-cuti:end]    # MSQR + ADAM + Qdiff
    mres2qConv = mres2q[end-cuti:end]    # MSQR + CONS + Qdiff
    sres1qConv = sres1q[end-cuti:end]    # SWAP + ADAM + Qdiff
    sres2qConv = sres2q[end-cuti:end]    # SWAP + CONS + Qdiff
    mres1nConv = mres1n[end-cuti:end]    # MSQR + ADAM + Ndiff
    mres2nConv = mres2n[end-cuti:end]    # MSQR + CONS + Ndiff
    sres1nConv = sres1n[end-cuti:end]    # SWAP + ADAM + Ndiff
    sres2nConv = sres2n[end-cuti:end]    # SWAP + CONS + Ndiff
    # @show mres1n_2Conv = mres1n_2[end-cuti:end]
    # @show sres1n_2Conv = sres1n_2[end-cuti:end]
    mres1n_2Conv = mres1n_2[end-cuti:end]
    sres1n_2Conv = sres1n_2[end-cuti:end]
    
    mres1qMean = mres1qConv |> mean
    mres2qMean = mres2qConv |> mean
    sres1qMean = sres1qConv |> mean
    sres2qMean = sres2qConv |> mean
    mres1nMean = mres1nConv |> mean
    mres2nMean = mres2nConv |> mean
    sres1nMean = sres1nConv |> mean
    sres2nMean = sres2nConv |> mean
    mres1n_2Mean = mres1n_2Conv |> mean
    sres1n_2Mean = sres1n_2Conv |> mean

    @test std(mres1qConv) ≤ tol*mres1qMean
    @test std(mres2qConv) ≤ tol*mres2qMean
    @test std(sres1qConv) ≤ tol*sres1qMean
    @test std(sres2qConv) ≤ tol*sres2qMean
    @test std(mres1nConv) ≤ tol*mres1nMean
    @test std(mres2nConv) ≤ tol*mres2nMean
    @test std(sres1nConv) ≤ tol*sres1nMean
    @test std(sres2nConv) ≤ tol*sres2nMean 

    # If ADAM and default SGD converge to the approximately same value.
    ## (Including Ndiff and Qdiff situations)
    tol2 = 0.06
    @test isapprox(mres1qMean, mres2qMean, atol=tol2*max( mres1qMean, mres2qMean ))
    @test isapprox(sres1qMean, sres2qMean, atol=tol2*max( sres1qMean, sres2qMean ))
    @test isapprox(mres1nMean, mres2nMean, atol=tol2*max( mres1nMean, mres2nMean ))
    @test isapprox(sres1nMean, sres2nMean, atol=tol2*max( sres1nMean, sres2nMean ))

    # If MSQRtrain and SWAPtrain converge to the approximately same value.
    ## (Including Ndiff and Qdiff situations)
    tol3 = 0.06
    @test isapprox(mres1qMean, sres1qMean, atol=tol3*max( mres1qMean, sres1qMean ))
    @test isapprox(mres2qMean, sres2qMean, atol=tol3*max( mres2qMean, sres2qMean ))
    @test isapprox(mres1nMean, sres1nMean, atol=tol3*max( mres1nMean, sres1nMean ))
    @test isapprox(mres2nMean, sres2nMean, atol=tol3*max( mres2nMean, sres2nMean ))

    # If the auto-train actually converges to theoretical maximum.
    @test isapprox(mres1n_2Mean, 1.0, atol = 1.5e-2)
    @test isapprox(sres1n_2Mean, 1.0, atol = 1.5e-2)

    # If the training curves monotonically rise.
    function trendCompr(res::Array{Float64, 1})
        midv = middle(res)
        @test res[end] > midv > res[1]
        max = (2midv-res[1])
        @test res[end] / max > 0.9 
    end
    trendCompr(mres1q)
    trendCompr(sres1q)
    trendCompr(mres2q)
    trendCompr(sres2q)
    trendCompr(mres1n)
    trendCompr(sres1n)
    trendCompr(mres2n)
    trendCompr(sres2n)

    # If the overlaps from MSQR have enough accuracy.
    function oltest(regT::ArrayReg, cGen::ChainBlock, nM::Int64, ol0::Float64)
        nq = nqubits(cGen)
        regG = zero_state(nq, nbatch=nM) |> cGen
        ol = ((regT.state'*regG.state)[1] |> abs)^2
        @test isapprox(ol, ol0, atol=0.005) 
    end
    oltest(regT, mps_m1q.cExtend, m, mres1q[end])
    oltest(regT, c_s1q, m, sres1q[end])
    oltest(regT, mps_m2q.cExtend, m, mres2q[end])
    oltest(regT, c_s2q, m, sres2q[end])
    oltest(regT, mps_m1n.cExtend, m, mres1n[end])
    oltest(regT, c_s1n, m, sres1n[end])
    oltest(regT, mps_m2n.cExtend, m, mres2n[end])
    oltest(regT, c_s2n, m, sres2n[end])
end