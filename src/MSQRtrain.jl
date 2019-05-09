#=
Combining MPS-SwapTest method and Quantum Gradient Optimization, 
train a MPS circuit with adjustable parameters to recreate target wave function.
=#
export MSQRtrain!, GDescent


"""
    MSQRtrain!(regTar::DefaultRegister, MSCircuit::ChainBlock, nMeasure::Int64, nTrain::Int64; Gmethod::String="Qdiff", GDmethod=("default",0.01), show::Bool=false) -> overlaps::Array{Float64,1}
MSQR training function. This function will change the parameters of differentiable gates in MSCircuit.
"""
function MSQRtrain!(regTar::DefaultRegister, MSCircuit::ChainBlock, nMeasure::Int64, nTrain::Int64;
                    Gmethod::String="Qdiff", GDmethod=("default",0.01), show::Bool=false)
    nbset = nBitSet(MSCircuit)
    nBitT = nbset.nBitT
    vBit = nbset.vBit
    rBit = nbset.rBit
    if show
        println("\nMSQR Training Parameters:")
        println("nBitT=$(nBitT) vBit=$(vBit) rBit=$(rBit)")
        println("nMeasure=$(nMeasure) nTrain=$(nTrain) GDmethod=$(GDmethod)\n")
        println("Initial overlap = $(MStest(regTar, MSCircuit, nMeasure).overlap)")
    end
    res = train!(nTrain, MSCircuit, Tmethod = MSCircuit->MStest(regTar, MSCircuit, nMeasure), 
                 Gmethod=Gmethod, GDmethod=GDmethod, show=show)
end


"""
    train!(nTrain::Int64, circuit0::ChainBlock; Tmethod::Function, Gmethod::String="Qdiff", GDmethod=("default",0.01), show::Bool=false) -> overlaps::Array{Float64,1}
Function that trains parameters of differentiable gates in the input circuit with designated overlap method 
in order to make the circuit generate a similar wave function to the target wave function. 
"""
function train!(nTrain::Int64, circuit0::ChainBlock; Tmethod::Function, 
                Gmethod::String="Qdiff", GDmethod=("default",0.01), show::Bool=false)
    overlaps = Float64[]
    GD = GDescent(GDmethod)
    tm=Tmethod(circuit0)
    witnessOp = matblock(diagm(0=>ones(1<<nqubits(tm.witnessOp))) - mat(tm.witnessOp))
    circuit = circuit0
    dGates = collect_blocks(AbstractDiff, circuit)
    for i=1:nTrain
        if Gmethod == "Qdiff"
            grads =  getQdiff.(()->Tmethod(circuit).reg, dGates, Ref(witnessOp))
        elseif Gmethod == "Ndiff"
            grads = -getNdiff.(()->Tmethod(circuit).overlap, dGates, δ=0.005)
        end 
        dGatesPar = [parameters(dGates[i])[1] for i=1:length(dGates)]
        dGatesPar = GD(dGatesPar, grads) 
        dispatch!.(dGates, dGatesPar)
        tm = Tmethod(circuit)
        showCase = (25*i)%nTrain==0
        if  showCase && show
            println("Training Step $(i), overlap = $(tm.overlap)")
        end
        push!(overlaps,tm.overlap)
    end
    overlaps
end


"""
    GDescent(GDmethod::Union{String, Tuple{String,Float64}, Tuple{String,Float64,Tuple{Float64,Float64}}}) -> reF::Function
Function that returns the aimed Gradient Descent method.
"""
function GDescent(GDmethod::Union{String, Tuple{String,Float64}, Tuple{String,Float64,Tuple{Float64,Float64}}})
    if GDmethod == "ADAM"
        resF = (dGatesPar, grads) -> Optimise.update!(ADAM(), dGatesPar, grads)
    elseif GDmethod[1] == "ADAM"
        if length(GDmethod) == 2
            resF = (dGatesPar, grads) -> Optimise.update!(ADAM(GDmethod[2]), dGatesPar, grads)
        elseif length(GDmethod) == 3    
            resF = (dGatesPar, grads) -> Optimise.update!(ADAM(GDmethod[2],GDmethod[3]), dGatesPar, grads)
        end    
    elseif GDmethod[1] == "default" && length(GDmethod) == 2
        resF = (dGatesPar, grads) -> dGatesPar - grads.*GDmethod[2]
    end
    resF
end