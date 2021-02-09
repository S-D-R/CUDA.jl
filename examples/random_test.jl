using CUDA
using Cthulhu

N = 8

function kernel(A)
    tid = threadIdx().x
    r = CUDA.rand_dev()
    A[tid] = r
    return nothing
end

a = ones(Int64, N)
dev_a = CuArray(a)

#= @device_code_warntype interactive=false =# @cuda threads=N kernel(dev_a)

println(Array(dev_a))
