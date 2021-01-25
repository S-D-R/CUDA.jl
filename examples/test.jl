using Revise
using IRTools
using CUDA

const N = 8
const const_mem = CuConstantMemory(collect(Int32, 1:N))

struct CallableStruct
    x::Int32
end

(c::CallableStruct)(y) = c.x + y

const callable_struct = CallableStruct(7)

function kernel(A)
    tid = threadIdx().x
    x = callable_struct(tid)
    A[tid] = const_mem[tid] + x
    return nothing
end

A = collect(Int32, 1:N)

ci = @code_typed kernel(A)

ci = ci[1]

for x in ci.code
    if x isa GlobalRef
        val = getproperty(x.mod, x.name)
        val_converted = cudaconvert(val)
        val === val_converted && continue
        println("$x: $val => $val_converted")
    end
end

ir = @code_ir kernel(A)

for st in ir
    i, st = st
    expr = st.expr
    println(i)
    println(st)
    println(expr)
    println(expr.head, "====", expr.args)
end
