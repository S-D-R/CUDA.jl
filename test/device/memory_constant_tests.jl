@testset "constant memory" begin
    N = 8
    
    @testset "basic" begin
        init = ones(Float32, N)
        const_mem = CuConstantMemory(init)

        function kernel(a::CuDeviceArray{Float32})
            tid = threadIdx().x
        
            a[tid] = const_mem[tid]
        
            return nothing
        end

        a = zeros(Float32, N)
        dev_a = CuArray(a)

        @cuda threads = N kernel(dev_a)

        @test Array(dev_a) == init
    end

    @testset "2d constant memory" begin
        init = ones(Float32, N, N)
        const_mem = CuConstantMemory(init)

        function kernel(a::CuDeviceArray{Float32})
            i = blockIdx().x
            j = blockIdx().y
            
            a[i, j] = const_mem[i, j]
    
            return nothing
        end

        a = zeros(Float32, N, N)
        dev_a = CuArray(a)

        @cuda blocks = (N, N) kernel(dev_a)

        @test Array(dev_a) == init
    end

    @testset "reuse between kernels" begin
        init = ones(Int32, N)
        const_mem = CuConstantMemory(init)
        
        function kernel1(a::CuDeviceArray{Int32})
            tid = threadIdx().x

            a[tid] += const_mem[tid]

            return nothing
        end

        function kernel2(b::CuDeviceArray{Int32})
            tid = threadIdx().x
            
            b[tid] -= const_mem[tid]

            return nothing
        end

        a = ones(Int32, N)
        b = ones(Int32, N)
        dev_a = CuArray(a)
        dev_b = CuArray(b)

        @cuda threads = N kernel1(dev_a)
        @cuda threads = N kernel2(dev_b)

        @test Array(dev_a) == a + init
        @test Array(dev_b) == b - init
    end

    @testset "mutation" begin
        init = ones(Float32, N)
        const_mem = CuConstantMemory(init)

        function kernel(a::CuDeviceArray{Float32})
            tid = threadIdx().x
    
            a[tid] = const_mem[tid]
    
            return nothing
        end

        a = zeros(Float32, N)
        dev_a = CuArray(a)
        
        @cuda threads = N kernel(dev_a)
        
        @test Array(dev_a) == init

        new_value = collect(Float32, 1:N)
        copyto!(const_mem, new_value)
        
        @cuda threads = N kernel(dev_a)
        
        @test Array(dev_a) == new_value

        @test_throws DimensionMismatch copyto!(const_mem, ones(Float32, N - 1))
    end
end
