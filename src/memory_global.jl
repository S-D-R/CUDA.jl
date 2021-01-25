export CuGlobalMemory

# Map a global memory name to its array value
const global_memory_initializer = Dict{Symbol,WeakRef}()

struct CuGlobalMemory{T,N} <: AbstractArray{T,N}
    name::Symbol
    value::Array{T,N}

    function CuGlobalMemory(value::Array{T,N}) where {T,N}
        # TODO: add finalizer that removes the relevant entry from global_memory_initializer?
        Base.isbitstype(T) || throw(ArgumentError("CuGlobalMemory only supports bits types"))
        name = gensym("global_memory")
        name = GPUCompiler.safe_name(string(name))
        name = Symbol(name)
        val = deepcopy(value)
        global_memory_initializer[name] = WeakRef(val)
        return new{T,N}(name, val)
    end
end

CuGlobalMemory{T}(::UndefInitializer, dims::Integer...) where {T} =
    CuGlobalMemory(Array{T}(undef, dims))
CuGlobalMemory{T}(::UndefInitializer, dims::Dims{N}) where {T,N} =
    CuGlobalMemory(Array{T,N}(undef, dims))

Base.size(A::CuGlobalMemory) = size(A.value)

Base.getindex(A::CuGlobalMemory, i::Integer) = Base.getindex(A.value, i)
Base.setindex!(A::CuGlobalMemory, v, i::Integer) = Base.setindex!(A.value, v, i)
Base.IndexStyle(::Type{<:CuGlobalMemory}) = Base.IndexLinear()

Adapt.adapt_storage(::Adaptor, A::CuGlobalMemory{T,N}) where {T,N} = 
    CuDeviceGlobalMemory{T,N,A.name,size(A.value)}()


function Base.copyto!(global_mem::CuGlobalMemory{T}, value::Array{T}, kernel::HostKernel) where T
    # TODO: add bool argument to also change the value field of global_mem?
    if size(global_mem) != size(value)
        throw(DimensionMismatch("size of `value` does not match size of global memory"))
    end

    global_array = CuGlobalArray{T}(kernel.mod, string(global_mem.name), length(global_mem))
    copyto!(global_array, value)
end


function emit_global_memory_initializer!(mod::LLVM.Module)
    for global_var in globals(mod)
        T_global = llvmtype(global_var)
        if addrspace(T_global) == AS.Global
            global_memory_name = Symbol(LLVM.name(global_var))
            if !haskey(global_memory_initializer, global_memory_name)
                continue # non user defined global memory, most likely from the CUDA runtime
            end

            arr = global_memory_initializer[global_memory_name].value
            @assert !isnothing(arr) "calling kernel containing garbage collected global memory"

            flattened_arr = reduce(vcat, arr)
            ctx = LLVM.context(mod)
            typ = eltype(eltype(T_global))

            # TODO: have a look at how julia converts structs to llvm:
            #       https://github.com/JuliaLang/julia/blob/80ace52b03d9476f3d3e6ff6da42f04a8df1cf7b/src/cgutils.cpp#L572
            #       this only seems to emit a type though
            if isa(typ, LLVM.IntegerType) || isa(typ, LLVM.FloatingPointType)
                init = ConstantArray(flattened_arr, ctx)
            elseif isa(typ, LLVM.ArrayType) # a struct with every field of the same type gets optimized to an array
                constant_arrays = LLVM.Constant[]
                for x in flattened_arr
                    fields = collect(map(name->getfield(x, name), fieldnames(typeof(x))))
                    constant_array = ConstantArray(fields, ctx)
                    push!(constant_arrays, constant_array)
                end
                init = ConstantArray(typ, constant_arrays)
            elseif isa(typ, LLVM.StructType)
                constant_structs = LLVM.Constant[]
                for x in flattened_arr
                    constants = LLVM.Constant[]
                    for fieldname in fieldnames(typeof(x))
                        field = getfield(x, fieldname)
                        if isa(field, Bool)
                            # NOTE: Bools get compiled to i8 instead of the more "correct" type i1
                            push!(constants, ConstantInt(LLVM.Int8Type(ctx), field))
                        elseif isa(field, Integer)
                            push!(constants, ConstantInt(field, ctx))
                        elseif isa(field, AbstractFloat)
                            push!(constants, ConstantFP(field, ctx))
                        else
                            throw(error("global memory does not currently support structs with non-primitive fields ($(typeof(x)).$fieldname::$(typeof(field)))"))
                        end
                    end
                    const_struct = ConstantStruct(typ, constants)
                    push!(constant_structs, const_struct)
                end
                init = ConstantArray(typ, constant_structs)
            else
                # unreachable, but let's be safe and throw a nice error message just in case
                throw(error("could not emit initializer for global memory of type $typ"))
            end
            
            initializer!(global_var, init)
        end
    end
end
