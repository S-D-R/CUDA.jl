using Random
using Adapt

mutable struct RNG
    seed::CuArray{Int64,1}
    m::Int64
    a::Int64
    c::Int64
    function RNG(seed::Int64=rand(Random.RandomDevice(), Int64), m::Int64=2^48, a::Int64=25214903917, c::Int64=11)
        new(CuArray([seed]), m, a, c)
    end
end

struct DeviceRNG
    seed::CuDeviceArray{Int64,1,AS.Global}
    m::Int64
    a::Int64
    c::Int64
end

Adapt.adapt_storage(to::CUDA.Adaptor, rng::RNG) = DeviceRNG(adapt(to, rng.seed), rng.m, rng.a, rng.c)

# LCG: https://en.wikipedia.org/wiki/Linear_congruential_generator
function Base.rand(rng::DeviceRNG)
    # TODO: atomics, mutex?
    rng.seed[1] = (rng.a * rng.seed[1] + rng.c) % rng.m
end

function test_rand()
    N = 16

    rng = RNG()

    @show rng

    dump(rng)
    
    function kernel(A)
        tid = threadIdx().x
        @cushow rng.seed[1]
        A[tid] = rand(rng)
        @cushow rng.seed[1]

        return nothing
    end

    a = zeros(Int64, N)
    dev_a = CuArray(a)

    @cuda threads=N kernel(dev_a)

    println(Array(dev_a))
end

test_rand()

