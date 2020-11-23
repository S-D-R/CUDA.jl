# Constant Memory

export CuConstantMemory, initialize_constant_memory

"""
    CuConstantMemory{T,N,Name,Shape}(value::Array{T,N})

Construct an `N`-dimensional constant memory array of type `T`. The `Name` and `Shape` type parameters 
are implementation details and it is discouraged to use them directly, instead use 
[name(::CuConstantMemory)](@ref) and [Base.size(::CuConstantMemory)](@ref) respectively.

`CuConstantMemory` is read-only, and reads are only defined within the context of a CUDA kernel function. 
Attempts to read from `CuConstantMemory` outside of a kernel function will lead to undefined behavior.

Note that `deepcopy` will be called on the `value` constructor argument, meaning that mutations to `value` or its
elements after construction will not be reflected in the value of `CuConstantMemory`. If mutation of `CuConstantMemory`
is desired, please use [function Base.copyto!(constant_memory::CuConstantMemory{T}, src::Array{T})](@ref)

Unlike in CUDA C, structs cannot be put directly into constant memory, but this can be emulated by
simply wrapping your struct inside of a 1-element array.
"""
struct CuConstantMemory{T,N,Name,Shape} <: AbstractArray{T,N}
    function CuConstantMemory(value::Array{T,N}) where {T,N}
        Name = gensym("constant_memory")
        Name = GPUCompiler.safe_name(string(Name))
        Name = Symbol(Name)
        Shape = size(value)
        t = new{T,N,Name,Shape}()
        constant_memory_init_dict[t] = deepcopy(value)
        return t
    end
end

"""
Get the name of underlying global variable of this `CuConstantMemory`
"""
name(::CuConstantMemory{T,N,Name,Shape}) where {T,N,Name,Shape} = Name

Base.:(==)(a::CuConstantMemory, b::CuConstantMemory) = name(a) == name(b)
Base.hash(a::CuConstantMemory, h::UInt) = hash(name(a), h)

Base.size(::CuConstantMemory{T,N,Name,Shape}) where {T,N,Name,Shape} = Shape

Base.show(io::IO, A::CuConstantMemory) = print(io, "$(typeof(A))")

Base.show(io::IO, ::MIME"text/plain", A::CuConstantMemory) = show(io, A)

Base.@propagate_inbounds Base.getindex(A::CuConstantMemory, i::Integer) = constmemref(A, i)

Base.IndexStyle(::Type{<:CuConstantMemory}) = Base.IndexLinear()

# TODO: ideally we would use WeakKeyDict here as to not use up memory unnessecarily,
#       but we can't due to the isbits requirement of CuConstantMemory
#       perhaps we can use a WeakRef directly as the key?
const constant_memory_init_dict = Dict{CuConstantMemory,Array}()

function Base.copyto!(constant_memory::CuConstantMemory{T}, src::Array{T}) where T
    if size(constant_memory) != size(src)
        throw(DimensionMismatch("size of `src` does not match size of constant memory"))
    end
    constant_memory_init_dict[constant_memory] = src
end

"""
Initialize all constant memory in the given `mod`.
"""
function initialize_constant_memory(mod::CuModule)
    for (constant_memory, array) in constant_memory_init_dict
        try
            global_array = CuGlobalArray{eltype(constant_memory)}(mod, string(name(constant_memory)), length(constant_memory))
            copyto!(global_array, array)
        catch e
            if isa(e, CuError) && e.code == CUDA_ERROR_NOT_FOUND
                # this `constant_memory` does not occur in the given `mod`
                continue
            else
                throw(e)
            end
        end
    end
end

@inline function constmemref(A::CuConstantMemory{T,N,Name,Shape}, index::Integer) where {T,N,Name,Shape}
    @boundscheck checkbounds(A, index)
    len = length(A)
    return read_constant_mem(Val(Name), index, T, Val(len))
end

@generated function read_constant_mem(::Val{global_name}, index::Integer, ::Type{T}, ::Val{len}) where {global_name,T,len}
    JuliaContext() do ctx
        # define LLVM types
        T_int = convert(LLVMType, Int, ctx)
        T_result = convert(LLVMType, T, ctx)

        # define function and get LLVM module
        param_types = [T_int]
        llvm_f, _ = create_function(T_result, param_types)
        mod = LLVM.parent(llvm_f)

        # create a constant memory global variable
        global_name_string = string(global_name)
        T_global = LLVM.ArrayType(T_result, len)
        global_var = GlobalVariable(mod, T_global, global_name_string, AS.Constant)
        initializer!(global_var, null(T_global))
        extinit!(global_var, true)
        # TODO: global_var alignment?

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            T_global_ptr = LLVM.PointerType(T_global)
            global_var_ptr = addrspacecast!(builder, global_var, T_global_ptr)
            typed_ptr = inbounds_gep!(builder, global_var_ptr, [ConstantInt(0, ctx), parameters(llvm_f)[1]])
            ld = load!(builder, typed_ptr)

            metadata(ld)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Constant, ctx)

            ret!(builder, ld)
        end

        # call the function
        call_function(llvm_f, T, Tuple{Int}, :((Int(index - one(index))),))
    end
end
