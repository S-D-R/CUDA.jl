using CUDA
using Cthulhu

N = 8

function kernel(A)
    tid = threadIdx().x
    r = CUDA.rand()
    A[tid] = r + CUDA.rand()
    return nothing
end

a = ones(Int64, N^2)
dev_a = CuArray(a)

@device_code dir="asm" @cuda threads=N kernel(dev_a)

println(Array(dev_a))
