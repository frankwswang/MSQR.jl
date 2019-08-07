# push!(LOAD_PATH, abspath("./src"))
# push!(LOAD_PATH, abspath("../MPSCircuit.jl/src"))
using Test, MSQR, MPSCircuit, Yao, Random, Statistics, Yao.ConstGate
# Using CuYao to test GPU acceleration compatibility.
using CuYao

seedNum = 1234

@testset "SWAPtest(SWAPtrain.jl)" begin
    m = 5000
    n = 3
    Random.seed!(seedNum)
    cgen = dispatch!(DCbuilder(n,4).body, :random) 
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
    m = 5000 
    n = 6
    Random.seed!(seedNum)
    regT = rand_state(n)

    v = 1
    r = 1
    Random.seed!(seedNum)
    mps1 = MPSC("CS", n, v, r)
    c1 = MScircuit(n, v, r, mps1.cBlocks)
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

    v = 2
    r = 1
    d = 3
    Random.seed!(seedNum)
    mps2 = MPSC(("DC",d), n, v, r)
    c2 = MScircuit(n, v, r, mps2.cBlocks)
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

@testset "MSQRtrain.jl + SWAPtrain!(SWAPtrain.jl)" begin
    n = 4
    v = 2
    r = 1
    d = 2
    Random.seed!(seedNum)
    regT = rand_state(n)
    Random.seed!(seedNum)
    mps = MPSC(("DC",d),n,v,r)
    c = MScircuit(n, v, r, mps.cBlocks)
    md1 = ("ADAM", 0.05)
    md2 = ("default", 0.2)
    nT1 = 150
    nT2 = 300 
    m = 500

    mps_m1q = deepcopy(mps)
    c_m1q = MScircuit(n, v, r, mps_m1q.cBlocks)
    mps_s1q = deepcopy(mps)
    c_s1q = mps_s1q.cExtend
    mps_m2q = deepcopy(mps)
    c_m2q = MScircuit(n, v, r, mps_m2q.cBlocks)
    mps_s2q = deepcopy(mps)
    c_s2q = mps_s2q.cExtend
    mps_m1n = deepcopy(mps)
    c_m1n = MScircuit(n, v, r, mps_m1n.cBlocks)
    mps_s1n = deepcopy(mps)
    c_s1n = mps_s1n.cExtend
    mps_m2n = deepcopy(mps)
    c_m2n = MScircuit(n, v, r, mps_m2n.cBlocks)
    mps_s2n = deepcopy(mps)
    c_s2n = mps_s2n.cExtend

    Random.seed!(seedNum)
    mres1q = MSQRtrain!(regT, c_m1q, nT1, nMeasure=m, GDmethod = md1)
    Random.seed!(seedNum)
    sres1q = SWAPtrain!(regT, c_s1q, nT1, nMeasure=m, GDmethod = md1)
    Random.seed!(seedNum)
    mres2q = MSQRtrain!(regT, c_m2q, nT2, nMeasure=m, GDmethod = md2)
    Random.seed!(seedNum)
    sres2q = SWAPtrain!(regT, c_s2q, nT2, nMeasure=m, GDmethod = md2)
    Random.seed!(seedNum)
    mres1n = MSQRtrain!(regT, c_m1n, nT1, nMeasure=m, GDmethod = md1, Gmethod="Ndiff")
    Random.seed!(seedNum)
    sres1n = SWAPtrain!(regT, c_s1n, nT1, nMeasure=m, GDmethod = md1, Gmethod="Ndiff")
    Random.seed!(seedNum)
    mres2n = MSQRtrain!(regT, c_m2n, nT2, nMeasure=m, GDmethod = md2, Gmethod="Ndiff")
    Random.seed!(seedNum)
    sres2n = SWAPtrain!(regT, c_s2n, nT2, nMeasure=m, GDmethod = md2, Gmethod="Ndiff")

    tol2 = 0.05
    @test isapprox.(mean(mres1q[end-9,end]), mean(mres2q[end-9,end]), atol=tol2)
    @test isapprox.(mean(sres1q[end-9,end]), mean(sres2q[end-9,end]), atol=tol2)
    @test isapprox.(mean(mres1n[end-9,end]), mean(mres2n[end-9,end]), atol=tol2)
    @test isapprox.(mean(sres1n[end-9,end]), mean(sres2n[end-9,end]), atol=tol2)

    tol3 = 0.05
    @test isapprox.(mean(mres1q[end-9,end]), mean(sres1q[end-9,end]), atol=tol3)
    @test isapprox.(mean(mres2q[end-9,end]), mean(sres2q[end-9,end]), atol=tol3)
    @test isapprox.(mean(mres1n[end-9,end]), mean(sres1n[end-9,end]), atol=tol3)
    @test isapprox.(mean(mres2n[end-9,end]), mean(sres2n[end-9,end]), atol=tol3)

    function trendCompr(res::Array{Float64, 1})
        midv = middle(res)
        @test res[end] > midv > res[1]
        max = (2midv-res[1])
        @test res[end] / max > 0.8 
    end
    trendCompr(mres1q)
    trendCompr(sres1q)
    trendCompr(mres2q)
    trendCompr(sres2q)
    trendCompr(mres1n)
    trendCompr(sres1n)
    trendCompr(mres2n)
    trendCompr(sres2n)

    function oltest(regT::DefaultRegister, cGen::ChainBlock, nM::Int64, ol0::Float64)
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