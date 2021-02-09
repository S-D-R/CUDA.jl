const m = Int64(2^48)
const a = Int64(25214903917)
const c = Int64(11)

function rand_dev()
    bx, by, bz = blockIdx()
    tx, ty, tz = threadIdx()

    seed = read_random_mem(bx, by, bz, tx, ty, tz, Int64)

    # LCG: https://en.wikipedia.org/wiki/Linear_congruential_generator
    new_seed = (a * seed + c) % m

    write_random_mem(bx, by, bz, tx, ty, tz, new_seed, Int64)

    return new_seed
end

@generated function read_random_mem(bx::Int64, by::Int64, bz::Int64, tx::Int64, ty::Int64, tz::Int64, ::Type{T}) where {T}
    JuliaContext() do ctx
        # define LLVM types
        T_result = convert(LLVMType, T, ctx)

        # define function and get LLVM module
        param_types = []
        llvm_f, _ = create_function(T_result, param_types)
        mod = LLVM.parent(llvm_f)

        # create a random memory global variable
        global_name = GPUCompiler.safe_name("cuda_random_number_$bx$by$(bz)_$tx$ty$tz")
        global_var = GlobalVariable(mod, T_result, global_name, AS.Global)
        linkage!(global_var, LLVM.API.LLVMWeakAnyLinkage) # merge, but make sure symbols aren't discarded
        initializer!(global_var, null(T_result))
        extinit!(global_var, true)
        
        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            typed_ptr = inbounds_gep!(builder, global_var, [ConstantInt(0, ctx), ConstantInt(0, ctx)])
            ld = load!(builder, typed_ptr)

            metadata(ld)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Global, ctx)

            ret!(builder, ld)
        end

        # call the function
        call_function(llvm_f, T, Tuple{})
    end
end


@generated function write_random_mem(bx::Int64, by::Int64, bz::Int64, tx::Int64, ty::Int64, tz::Int64, x::Int64, ::Type{T}) where {T}
    JuliaContext() do ctx
        # define LLVM types
        eltyp = convert(LLVMType, Int64, ctx)

        # define function and get LLVM module
        param_types = [eltyp]
        llvm_f, _ = create_function(LLVM.VoidType(ctx), param_types)
        mod = LLVM.parent(llvm_f)

        # create a random memory global variable
        global_name = GPUCompiler.safe_name("cuda_random_number_$bx$by$(bz)_$tx$ty$tz")
        global_var = GlobalVariable(mod, eltyp, string(global_name), AS.Global)
        linkage!(global_var, LLVM.API.LLVMWeakAnyLinkage) # merge, but make sure symbols aren't discarded
        initializer!(global_var, null(eltyp))
        extinit!(global_var, true)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            typed_ptr = inbounds_gep!(builder, global_var, [ConstantInt(0, ctx), ConstantInt(0, ctx)])
            val = parameters(llvm_f)[1]
            st = store!(builder, val, typed_ptr)

            metadata(st)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Global, ctx)

            ret!(builder)
        end

        # call the function
        call_function(llvm_f, Cvoid, Tuple{Int64}, :((x,)))
    end
end
