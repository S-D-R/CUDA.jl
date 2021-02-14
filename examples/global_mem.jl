using CUDA

function global_memory_test()
    N = 4
    global_mem = CuGlobalMemory(collect(Float32, 1:N))

    function kernel1_dynamic(tid)
        global_mem[tid] += 1
        return nothing
    end

    function kernel1(a::CuDeviceArray{Float32})	
        tid = threadIdx().x

        global_mem[tid] += 1

        #=
        @cuda dynamic=true kernel1_dynamic(tid)

        sync_threads()

        CUDA.device_synchronize()

        sync_threads()
        =#

        global_mem[tid] += 2

        a[tid] = global_mem[tid]

        return nothing
    end

    a = zeros(Float32, N)
    dev_a = CuArray(a)

    kernel = @cuda threads = N kernel1(dev_a)
    @show Array(dev_a)
    @show global_mem

    kernel(dev_a, threads=N)
    @show Array(dev_a)
    @show global_mem

    @device_code dir="asm" @cuda threads = N kernel1(dev_a)
    @show Array(dev_a)
    @show global_mem

    function kernel2(a::CuDeviceArray{Float32})	
        tid = threadIdx().x

        a[tid] = global_mem[tid]

        return nothing
    end

    @cuda threads = N kernel2(dev_a)
    @show Array(dev_a)
    @show global_mem

    return nothing
end

global_memory_test()
