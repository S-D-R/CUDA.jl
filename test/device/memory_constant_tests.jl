@testset "constant memory" begin
    N = 8

    @testset "array interface" begin
        init = ones(Int32, N)
        const_mem = CuConstantMemory(init)

        @test size(const_mem) == size(init)
        @test length(const_mem) == length(init)
    end
    
    @testset "basic" begin
        init = Int32[4, 5]
        const_mem = CuConstantMemory(init)

        function kernel(a::CuDeviceArray{Int32})
            tid = threadIdx().x

            a[tid] = const_mem[1] + const_mem[2]

            return nothing
        end

        a = zeros(Int32, N)
        dev_a = CuArray(a)
        
        @cuda threads = N kernel(dev_a)	

        @test all(Array(dev_a) .== sum(init))
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

    @testset "complex types" begin
        struct RGB{T}
            r::T
            g::T
            b::T
        end

        init = [RGB{Float32}(5.0, 7.0, 9.0), RGB{Float32}(6.0, 8.0, 10.0)]
        const_mem = CuConstantMemory(init)

        function kernel(a::CuDeviceArray{Float32}, b::CuDeviceArray{Float32})	
            tid = threadIdx().x

            a[tid] = const_mem[1].r + const_mem[2].g + const_mem[1].b
            b[tid] = const_mem[2].r + const_mem[1].g + const_mem[2].b

            return nothing
        end 

        a = zeros(Float32, N)
        b = zeros(Float32, N)
        dev_a = CuArray(a)
        dev_b = CuArray(b)

        @cuda threads = N kernel(dev_a, dev_b)

        @test all(Array(dev_a) .≈ init[1].r + init[2].g + init[1].b)
        @test all(Array(dev_b) .≈ init[2].r + init[1].g + init[2].b)

        @test_throws ArgumentError CuConstantMemory(["non", "isbits", "type", "isn't", "valid"])
    end

    @testset "inbounds" begin
        init = ones(Int32, N)
        const_mem = CuConstantMemory(init)

        function kernel(a::CuDeviceArray{Int32})	
            tid = threadIdx().x

            @inbounds a[tid] = const_mem[tid]

            return nothing
        end

        a = zeros(Int32, N)
        dev_a = CuArray(a)
        
        @cuda threads = N kernel(dev_a)	

        # TODO: check the generated code for bounds checks?
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
