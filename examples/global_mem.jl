using CUDA

function global_memory_test()
    N = 4
    global_mem = CuGlobalMemory(zeros(Float32, N))

    function kernel1_dynamic(tid)
        global_mem[tid] += 1
        return nothing
    end

    function kernel1(a::CuDeviceArray{Float32})	
        tid = threadIdx().x

        global_mem[tid] += 1

        @cuda dynamic=true kernel1_dynamic(tid)

        sync_threads()

        CUDA.device_synchronize()
        
        a[tid] = sum(global_mem)

        return nothing
    end

    a = zeros(Float32, N)
    dev_a = CuArray(a)

    kernel = @cuda threads = N kernel1(dev_a)
    println(Array(dev_a))

    kernel(dev_a, threads=N)
    println(Array(dev_a))

    # works due to caching
    @cuda threads = N kernel1(dev_a)
    println(Array(dev_a))

    function kernel2(a::CuDeviceArray{Float32})	
        tid = threadIdx().x

        a[tid] = sum(global_mem)

        return nothing
    end

    @cuda threads = N kernel2(dev_a)
    println(Array(dev_a))
end

global_memory_test()
