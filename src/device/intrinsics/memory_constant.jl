# Constant Memory

export CuConstantMemory

"""
    CuConstantMemory{T,N,Name,Shape}(value::Array{T,N})
    CuConstantMemory{T}(::UndefInitializer, dims::Dims{N})

Construct an `N`-dimensional constant memory array of type `T`, where `isbits(T)`. The `Name` 
and `Shape` type parameters are implementation details and it is discouraged to use them directly,
instead use [name(::CuConstantMemory)](@ref) and [Base.size(::CuConstantMemory)](@ref) respectively.

If the `UndefInitializer` constructor is used, all reads from `CuConstantMemory` will return `0`.
See `copyto!` for changing the value of an undef initialized `CuConstantMemory` object.

`CuConstantMemory` is read-only, and reads are only defined within the context of a CUDA kernel function. 
Attempts to read from `CuConstantMemory` outside of a kernel function will lead to undefined behavior.

Note that `deepcopy` will be called on the `value` constructor argument, meaning that mutations to `value`
or its elements after construction will not be reflected in the value of `CuConstantMemory`.
Mutation of `CuConstantMemory` is possible via `copyto!`.

Unlike in CUDA C, structs cannot be put directly into constant memory. This feature can be emulated however
by wrapping your struct inside of a 1-element array.
"""
struct CuConstantMemory{T,N,Name,Shape} <: AbstractArray{T,N}
    function CuConstantMemory(value::Array{T,N}) where {T,N}
        Base.isbitstype(T) || throw(ArgumentError("CuConstantMemory only supports bits types"))
        Name = gensym("constant_memory")
        Name = GPUCompiler.safe_name(string(Name))
        Name = Symbol(Name)
        Shape = size(value)
        t = new{T,N,Name,Shape}()
        constant_memory_initializer[Name] = deepcopy(value)
        return t
    end
    function CuConstantMemory{T}(::UndefInitializer, dims::Dims{N}) where {T,N}
        Base.isbitstype(T) || throw(ArgumentError("CuConstantMemory only supports bits types"))
        Name = gensym("constant_memory")
        Name = GPUCompiler.safe_name(string(Name))
        Name = Symbol(Name)
        Shape = dims
        return new{T,N,Name,Shape}()
    end
end

"""
Get the name of underlying global variable of this `CuConstantMemory`.
"""
name(::CuConstantMemory{T,N,Name,Shape}) where {T,N,Name,Shape} = Name

Base.:(==)(a::CuConstantMemory, b::CuConstantMemory) = name(a) == name(b)
Base.hash(a::CuConstantMemory, h::UInt) = hash(name(a), h)

Base.size(::CuConstantMemory{T,N,Name,Shape}) where {T,N,Name,Shape} = Shape

Base.show(io::IO, A::CuConstantMemory) = print(io, "$(typeof(A))")

Base.show(io::IO, ::MIME"text/plain", A::CuConstantMemory) = show(io, A)

Base.@propagate_inbounds Base.getindex(A::CuConstantMemory, i::Integer) = constmemref(A, i)

Base.IndexStyle(::Type{<:CuConstantMemory}) = Base.IndexLinear()

# FIXME: the array values in this dict will never be garbage collected,
#        WeakKeyDict doesn't work because of the isbits requirement of CuConstantMemory
const constant_memory_initializer = Dict{Symbol,Array}()

"""
Copy `value` into `const_mem`. Note that this will not change the value of `const_mem`
in kernels that are already compiled via `@cuda`.
"""
function Base.copyto!(const_mem::CuConstantMemory{T}, value::Array{T}) where T
    if size(const_mem) != size(value)
        throw(DimensionMismatch("size of `value` does not match size of constant memory"))
    end

    constant_memory_initializer[name(const_mem)] = value
end

"""
Given a `kernel` returned by `@cuda`, copy `value` into `const_mem`, both in this `kernel` and all 
new kernels using `const_mem`. If `const_mem` is not used within `kernel`, an error will be thrown.
"""
function Base.copyto!(kernel, const_mem::CuConstantMemory{T}, value::Array{T}) where T
    # FIXME: specifying kernel::HostKernel doesn't compile because the HostKernel type isn't defined yet
    if size(const_mem) != size(value)
        throw(DimensionMismatch("size of `value` does not match size of constant memory"))
    end

    constant_memory_initializer[name(const_mem)] = value

    global_array = CuGlobalArray{T}(kernel.mod, string(name(const_mem)), length(const_mem))
    copyto!(global_array, value)
end

function emit_constant_memory_initializer!(mod::LLVM.Module)
    for global_var in globals(mod)
        T_global = llvmtype(global_var)
        if addrspace(T_global) == AS.Constant
            constant_memory_name = Symbol(LLVM.name(global_var))
            if !haskey(constant_memory_initializer, constant_memory_name)
                # undef initializer, we trust the user to initialize manually
                return
            end
            arr = constant_memory_initializer[constant_memory_name]
            flattened_arr = reduce(vcat, arr)
            ctx = LLVM.context(mod)
            typ = eltype(eltype(T_global))
            # TODO: have a look at how julia converts structs to llvm:
            #       https://github.com/JuliaLang/julia/blob/80ace52b03d9476f3d3e6ff6da42f04a8df1cf7b/src/cgutils.cpp#L572
            #       this only seems to emit a type though
            if isa(typ, LLVM.IntegerType) || isa(typ, LLVM.FloatingPointType)
                init = ConstantArray(flattened_arr, ctx)
            elseif isa(typ, LLVM.ArrayType) # a struct with every field of the same type gets optimized to an array
                constant_arrays = ConstantArray[]
                for x in flattened_arr
                    fields = collect(map(name->getfield(x, name), fieldnames(typeof(x))))
                    constant_array = ConstantArray(fields, ctx)
                    push!(constant_arrays, constant_array)
                end
                init = ConstantArray(typ, constant_arrays)
            elseif isa(typ, LLVM.StructType)
                constant_structs = LLVM.ConstantStruct[]
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
                            throw(error("constant memory does not currently support structs with non-primitive fields ($(typeof(x)).$fieldname::$(typeof(field)))"))
                        end
                    end
                    const_struct = ConstantStruct(typ, constants)
                    push!(constant_structs, const_struct)
                end
                init = ConstantArray(typ, constant_structs)
            else
                # unreachable, but let's be safe and throw a nice error message just in case
                throw(error("could not emit initializer for constant memory of type $typ"))
            end
            initializer!(global_var, init)
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
        linkage!(global_var, LLVM.API.LLVMExternalLinkage) # NOTE: external linkage is the default
        extinit!(global_var, true)
        # TODO: global_var alignment?

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            typed_ptr = inbounds_gep!(builder, global_var, [ConstantInt(0, ctx), parameters(llvm_f)[1]])
            ld = load!(builder, typed_ptr)

            metadata(ld)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Constant, ctx)

            ret!(builder, ld)
        end

        # call the function
        call_function(llvm_f, T, Tuple{Int}, :((Int(index - one(index))),))
    end
end
