using Yao
using Yao.Blocks
using QuAlgorithmZoo
using LinearAlgebra
using Random
using Test

struct regMPSGen
    regA::DefaultRegister
    reg1::Array
    function regMPSGen(NBit::Int64)
        #println("s1")
        regA =zero_state(0)
        reg1= rand(MersenneTwister(1234), [0,1], NBit)
        #println("s2")
        for i=1:NBit
            reg = zero_state(1)
            reg1[i] == 1 && (reg |> X)
            #println("s3_1")
            regA = join(regA, reg)
            #println("s3_2")
        end
        #println("s4")
        new(regA, reg1)
    end
end

function RBit_SwapTest_circuit(nBitA::Int64, ϕ::Real)
    circuit = chain(nBitA, 
                    put(nBitA, nBitA=>H),
                    put(nBitA, nBitA=>shift(ϕ)),
                    chain(nBitA, [control(nBitA, nBitA, ( (nBitA-1),i )=>SWAP) for i=(nBitA-2):-1:1]),
                    put(nBitA, nBitA=>H)
                    )             
end


function MPS_SwapTest(regT::DefaultRegister, regG::regMPSGen, ϕ::Real)
    vBit = nqubits(regT)
    nBitA = vBit + 2
    circuit = RBit_SwapTest_circuit(nBitA, ϕ)
    #println("Step 1")
    regG1 = zero_state(1)
    regG.reg1[1] == 1 && (regG1 |> X)
    #println("Step 2")
    regA = join(zero_state(1), regG1)
    regA = join(regA, regT)
    #println("Step 2_2 regA: $(regA)")
    regA |> circuit[1] |> circuit[2]
    #println("Step 3 regA: \n $(regA.state)")
    for i = 1:(vBit-1)
        regA |> circuit[3][i]
        #println("Step 4_1")
        measure_reset!(regA, nBitA-1, val=regG.reg1[i+1])
        #println("Step 4_2")
    end
    regA |> circuit[3][vBit]
    #println("Step 5")
    regA |> circuit[4]
    res = expect(put(nBitA, nBitA=>Z), regA)
    #println("Step 6")
    res
end

@testset "Bit reusing Swap Test" begin
    NBit = 2
    ϕ = 0
    regTar = rand_state(NBit)
    regGen = regMPSGen(NBit)
    result = MPS_SwapTest(regTar, regGen, ϕ) |> tr

    rhoTar = regTar |> ρ
    rhoGen = regGen.regA |> ρ
    ExpectVal = tr(mat(rhoTar)*mat(rhoGen))
    result
    ExpectVal ≈ result
    @test  ExpectVal ≈ result   
end

# Test result is not ideal. Need to find the reason.