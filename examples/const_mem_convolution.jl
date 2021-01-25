using CUDA

const TILE_SIZE = 4
const INPUT_SIZE = 12
const MASK_WIDTH = 5

const M = CuConstantMemory(collect(Float32, 0:MASK_WIDTH - 1))

@inbounds function convolution_shared_memory(N::CuDeviceArray{Float32}, P::CuDeviceArray{Float32}, M::CuDeviceConstantMemory{Float32})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    N_s = @cuStaticSharedMem(Float32, TILE_SIZE)

	N_s[threadIdx().x] = N[i]

	sync_threads()

	this_title_start_point = (blockIdx().x - 1) * blockDim().x
	next_tile_start_point  = blockIdx().x * blockDim().x
    n_start_point = i - (MASK_WIDTH ÷ 2)
    Pvalue = 0.0

    for j = 0:MASK_WIDTH - 1
        N_index = n_start_point + j
        
        if N_index > 0 && N_index <= INPUT_SIZE
			if ((N_index > this_title_start_point) && (N_index <= next_tile_start_point))
                Pvalue += N_s[threadIdx().x + j - (MASK_WIDTH ÷ 2)] * M[j + 1]
    		else
				Pvalue += N[N_index] * M[j + 1];
            end
        end
    end

    P[i] = Pvalue
    
    return nothing
end

function run()
    start = time()

    N = collect(Float32, 0:INPUT_SIZE - 1)
    P = zeros(Float32, INPUT_SIZE)

    N_dev = CuArray(N)
    P_dev = CuArray(P)

    @cuda blocks = ((INPUT_SIZE + TILE_SIZE - 1) ÷ TILE_SIZE) threads = TILE_SIZE convolution_shared_memory(N_dev, P_dev, M)

    println(Array(P_dev))

    println("Elapsed: $(time() - start) seconds")
end

for _ in 1:5
    run()
    GC.gc(true)
end

CUDA.@profile CUDA.NVTX.@range "host" run()

#CUDA.synchronize()

#CUDA.@profile run()
