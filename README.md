# MSQR.jl
Qubit Replicate algorithm based on MPS-SwapTest hybrid method.

## Modules
### MPSSwapTest
Use Swap Test algorithm to measure overlaps between a Target wave function(register) and a random MPS wave function generated by a qubit-reusable circuit.

### MSQR
Combining MPS-Swap Test method and Quantum Gradient Optimization, MSQR can train a MPS circuit with adjustable parameters to recreate a wave function that has similar entanglments with the target wave function.

## Setup Guide
### Julia Environment
* [__Julia 1.1__](https://julialang.org)

### Installation
__Please first install another unregistered package [MPSCircuit](https://github.com/frankwswang/MPSCircuit.jl).__
Type `]` in Julia REPL to enter [`Pkg` mode](https://julialang.github.io/Pkg.jl/v1.0/index.html), then type:
```
pkg> add https://github.com/frankwswang/MPSCircuit.jl.git
```
__Then use the same approach to install this project package:__
```
pkg> add https://github.com/frankwswang/MSQR.jl.git
``` 
__ATTENTION:__ This packge is dependent on package [__Yao__](https://github.com/QuantumBFS/Yao.jl) and currently compatiple version is __Yao 0.4.1__. For the future development, you need to check its compatibility if you want to use it with a higher version of __Yao__. 

## Examples(How to use)
* __MSTest.jl:__ Showing the function of `MPSSwapTest`.
* __MSQRTest.jl:__ Showing a training example of `MSQR`.

## Reference
* Ekert, A. K., Alves, C. M., Oi, D. K., Horodecki, M., Horodecki, P., & Kwek, L. C. (2002). Direct estimations of linear and nonlinear functionals of a quantum state. Physical review letters, 88(21), 217901. ([DOI: 10.1103/PhysRevLett.88.217901](https://doi.org/10.1103/PhysRevLett.88.217901))

* Liu, J. G., Zhang, Y. H., Wan, Y., & Wang, L. (2019). Variational Quantum Eigensolver with Fewer Qubits. arXiv preprint, [arXiv:1902.02663](https://arxiv.org/abs/1902.02663). ([PDF](https://arxiv.org/pdf/1902.02663.pdf))

## License
MPSCircuit.jl is released under Apache License 2.0.