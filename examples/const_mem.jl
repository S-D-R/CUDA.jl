# TODO:
# shared memory
# vectorise?
# benchmarking 
#   => https://github.com/JuliaParallel/rodinia/blob/master/cuda/leukocyte/find_ellipse_kernel.cu
#   => https://github.com/JuliaParallel/rodinia/blob/master/julia_cuda/leukocyte/find_ellipse_kernel.jl
# struct support revisited (ConstantStruct)?
# immutable keyword? (constant propagation)
# structs with same fields?
# name clashing structs?
# better way to handle CuConstantMemory globals, currently we require globals to be passed as an argument (due to closure semantics)
#   => Rewrite code after compilation? cf. Traceur.jl for detecting globals
#   => Use StaticArrays.jl to make CuConstantMemory isbits?

using CUDA

function constant_memory_test()
    N = 4
    const_mem = CuConstantMemory(Float32[4, 5])

    function kernel(a::CuDeviceArray{Float32})	
        tid = threadIdx().x

        a[tid] = const_mem[1] + const_mem[2]

        return nothing
    end

    a = zeros(Float32, N)
    dev_a = CuArray(a)

    @cuda threads = N kernel(dev_a)

    println(Array(dev_a))
end

function constant_memory_test_2d()
    N = 4
    const_mem = CuConstantMemory(ones(Float32, N, N))

    function kernel(a::CuDeviceArray{Float32})
        i = blockIdx().x
        j = blockIdx().y
            
        a[i, j] = const_mem[i, j]
    
        return nothing
    end

    a = zeros(Float32, N, N)
    dev_a = CuArray(a)

    @cuda blocks = (N, N) kernel(dev_a)

    println(Array(dev_a))
end

struct RGB{T}
    r::T
    g::T
    b::T
end
 
function constant_memory_test_struct1()
    N = 4

    const_mem = CuConstantMemory([RGB{Float32}(5.0, 7.0, 9.0), RGB{Float32}(6.0, 8.0, 10.0)])

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

    println(Array(dev_a))
    println(Array(dev_b))
end

struct TestStruct
    x::Int32
    y::Float64
    z::Bool
end

function constant_memory_test_struct2()
    N = 4

    const_mem = CuConstantMemory([TestStruct(5, 6.0, true), TestStruct(7, 8.0, false)])

    function kernel(a::CuDeviceArray{Int32}, b::CuDeviceArray{Float32}, c::CuDeviceArray{Bool})	
        tid = threadIdx().x

        a[tid] = const_mem[1].x + const_mem[2].x
        b[tid] = const_mem[1].y + const_mem[2].y
        c[tid] = const_mem[1].z || const_mem[2].z

        return nothing
    end

    a = zeros(Int32, N)
    b = zeros(Float32, N)
    c = fill(false, N)
    dev_a = CuArray(a)
    dev_b = CuArray(b)
    dev_c = CuArray(c)

    @cuda threads = N kernel(dev_a, dev_b, dev_c)

    println(Array(dev_a))
    println(Array(dev_b))
    println(Array(dev_c))
end


function constant_memory_test_reuse()
    N = 4
    const_mem = CuConstantMemory(ones(Int32, N))
        
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

    println(Array(dev_a))
    println(Array(dev_b))

    @cuda threads = N kernel2(dev_b)

    println(Array(dev_a))
    println(Array(dev_b))
end

function constant_memory_test_mutation()
    N = 4

    const_mem = CuConstantMemory(ones(Float32, N))

    function kernel(a::CuDeviceArray{Float32})
        tid = threadIdx().x
    
        a[tid] = const_mem[tid]
    
        return nothing
    end

    a = zeros(Float32, N)
    dev_a = CuArray(a)
        
    kernel_obj = @cuda threads = N kernel(dev_a)

    println(dev_a)

    copyto!(const_mem, collect(Float32, 1:N), kernel_obj)

    kernel_obj(dev_a; threads=N)

    println(dev_a)
end

function constant_memory_test_undef_init()
    N = 4

    const_mem = CuConstantMemory{TestStruct}(undef, N, N)

    function kernel(a::CuDeviceArray{Int32})
        tid = threadIdx().x
    
        a[tid] = const_mem[tid].x
    
        return nothing
    end

    a = ones(Int32, N)
    dev_a = CuArray(a)
        
    @cuda threads = N kernel(dev_a)

    println(Array(dev_a))
end

function constant_memory_possible_bug()
    N = 8
    const_mem = CuConstantMemory{TestStruct}(undef, N)

    function kernel(a::CuDeviceArray{Int32})
        tid = threadIdx().x
    
        a[tid] = const_mem[tid].x
    
        return nothing
    end

    a = ones(Int32, N)
    dev_a = CuArray(a)

    kernel_obj = @cuda threads = N kernel(dev_a)

    println(Array(dev_a))
end

#= 
constant_memory_test()
constant_memory_test_2d()
constant_memory_test_struct1()
constant_memory_test_struct2()
constant_memory_test_reuse()
constant_memory_test_mutation()
constant_memory_test_undef_init()
constant_memory_possible_bug()
=#

const N = 8
const const_mem = CuConstantMemory(collect(1:N))

function kernel(a::CuDeviceArray{Int32})
    tid = threadIdx().x
    
    a[tid] = const_mem[tid]
    
    return nothing
end

a = ones(Int32, N)
dev_a = CuArray(a)

kernel_obj = @cuda threads = N kernel(dev_a)

println(Array(dev_a))