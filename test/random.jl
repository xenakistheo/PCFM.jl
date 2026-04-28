using CUDA, Lux
using NeuralOperators
using Random
using cuDNN

n_batch = 10
nz = 64
m_sample = cu(2 .+ randn(nz, 1, n_batch))

rng = Xoshiro()

fno2 = FourierNeuralOperator(gelu; chs=(1, 64, 64, 128, 1), modes=(16,))
ps,st = Lux.setup(rng, fno2) |> cu
val, st_ = Lux.apply(fno2, m_sample, ps, st)