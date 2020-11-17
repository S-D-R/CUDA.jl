using CUDA

const N = 8

const const_mem = CuConstantMemory(ones(Float32, N))
const const_mem_2d = CuConstantMemory(ones(Float32, N, N))

function kernel(a::CuDeviceArray{Float32}, b::CuDeviceArray{Float32})
    tid = threadIdx().x

    a[tid] = const_mem[tid]
    b[tid, tid] = const_mem_2d[tid, tid]

    return nothing
end

a = zeros(Float32, N)
b = zeros(Float32, N, N)
dev_a = CuArray(a)
dev_b = CuArray(b)

@device_code dir = "constant_assembly" @cuda threads = N kernel(dev_a, dev_b)

println(Array(dev_a))
display(Array(dev_b))
println()
