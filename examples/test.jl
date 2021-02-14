using CUDA

using CUDA

const N1 = 8
const N2 = 16

const const_mem1 = CuConstantMemory(fill(Int32(1), N1))
const const_mem2 = CuConstantMemory(fill(Int32(2), N2))

function kernel(A::CuDeviceArray, const_mem::CuDeviceConstantMemory)
    tid = threadIdx().x
    global_mem = CuDeviceGlobalMemory{Float32, 1, :test_global_memory, (N2,)}() # NOTE: possible to generically define device memory?
    global_mem[tid] = tid
    A[tid] = const_mem[tid] + global_mem[tid]
    return nothing
end

a = CuArray(zeros(Float32, N2))

println(Array(a))

@cuda threads=N1 kernel(a, const_mem1)

println(Array(a))

@cuda threads=N2 kernel(a, const_mem2)

println(Array(a))
