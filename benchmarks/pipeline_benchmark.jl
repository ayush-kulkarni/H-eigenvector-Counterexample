# Standalone performance investigation for the numerical pipeline's two hottest
# functions: kronecker_product and generate_symmetric_tensor (both defined in
# common/tensor_utils.jl). This script does NOT modify or replace either
# function; it imports them as-is and compares them against alternative
# implementations to characterize where the current approach spends time and
# memory. Run with:
#
#   julia --project=. benchmarks/pipeline_benchmark.jl

include(joinpath(@__DIR__, "..", "common", "tensor_utils.jl"))
using Combinatorics
using Statistics

# ---- Alternative implementations, used only for comparison ----

function alt_kronecker_product(A::AbstractArray{T,N}, B::AbstractArray{T,N}) where {T,N}
    sA, sB = size(A), size(B)
    outsize = ntuple(k -> sA[k] * sB[k], N)
    C = zeros(T, outsize)
    for idxA in CartesianIndices(A), idxB in CartesianIndices(B)
        outidx = ntuple(k -> (idxA[k] - 1) * sB[k] + idxB[k], N)
        C[outidx...] = A[idxA] * B[idxB]
    end
    return C
end

function alt_generate_symmetric_tensor(dim::Int, size_tuple::Tuple)
    tensor = zeros(size_tuple)
    seen = Dict{Vector{Int}, Float64}()
    for idx in CartesianIndices(size_tuple)
        key = sort(collect(Tuple(idx)))
        val = get!(seen, key) do
            rand(0.0:0.1:1.0)
        end
        tensor[idx] = val
    end
    return tensor
end

# ---- Correctness: alternative implementations must match the real pipeline ----

order, dimension = 3, 2   # matches the parameters used by run_check_on_random_tensors
size_tuple = ntuple(_ -> dimension, order)
A = generate_symmetric_tensor(order, size_tuple)
B = generate_symmetric_tensor(order, size_tuple)

@assert kronecker_product(B, A) == alt_kronecker_product(B, A) "Kronecker product outputs differ!"
println("Correctness check passed: alt_kronecker_product matches kronecker_product.")

# ---- Benchmark helper: run n times, record wall-clock ms and bytes allocated ----

function bench(f, args...; n=500)
    times = Float64[]
    bytes = Float64[]
    for _ in 1:n
        b = @allocated begin
            t0 = time_ns()
            f(args...)
            t1 = time_ns()
            push!(times, (t1 - t0) / 1e6)
        end
        push!(bytes, Float64(b))
    end
    return median(times), median(bytes)
end

kronecker_product(B, A); alt_kronecker_product(B, A)              # warm up (JIT)
t_current, b_current = bench(kronecker_product, B, A)
t_alt, b_alt = bench(alt_kronecker_product, B, A)

println("\n=== kronecker_product (order=$order, dimension=$dimension) ===")
println("Current (repeat + broadcast):     $(round(t_current, digits=4)) ms, $(Int(b_current)) bytes")
println("Alternative (preallocated loop):  $(round(t_alt, digits=4)) ms, $(Int(b_alt)) bytes")
println("Alternative is $(round(100*(1 - t_alt/t_current), digits=1))% faster, $(round(100*(1 - b_alt/b_current), digits=1))% less memory")

generate_symmetric_tensor(order, size_tuple); alt_generate_symmetric_tensor(order, size_tuple)  # warm up
t_gcur, b_gcur = bench(generate_symmetric_tensor, order, size_tuple)
t_galt, b_galt = bench(alt_generate_symmetric_tensor, order, size_tuple)

println("\n=== generate_symmetric_tensor (order=$order, dimension=$dimension) ===")
println("Current (Combinatorics-based):    $(round(t_gcur, digits=4)) ms, $(Int(b_gcur)) bytes")
println("Alternative (single-pass Dict):   $(round(t_galt, digits=4)) ms, $(Int(b_galt)) bytes")
println("Alternative is $(round(100*(1 - t_galt/t_gcur), digits=1))% faster, $(round(100*(1 - b_galt/b_gcur), digits=1))% less memory")
