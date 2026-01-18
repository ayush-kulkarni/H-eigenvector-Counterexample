using LinearAlgebra

"""
    check_heigenpair(A::Array{Float64, 4}, x::Vector{Complex{Float64}})

Verifies if a given pair (λ, x) is an H-eigenpair for a tensor A.
The function computes the candidate λ from the equation and checks consistency.

# Arguments
- `A`: The input tensor (assumed 4th order for this specific implementation, or generic?).
       The code loops 4 levels deep, so it seems specific to 4th order.
       Wait, the code says `m = ndims(A)` but the loops are nested: `for i,j,k,l`.
       If generic, it should use CartesianIndices or similar.
       The user provided code is specific to 4th order.
       However, the examples A and B are 3rd order (2x2x2).
       Let's look at `check_heigenpair` implementation again.
       The provided code in `numerical/h_eigenpair_verifier.jl` (read previously)
       had nested loops for i,j,k,l. This implies order 4.
       But A and B in `main.jl` are order 3.
       This verifier might be for another test case or needs to be generalized.
       
       Let's generalize it to order `m`.

# Returns
- `is_eigenvector::Bool`
- `is_normalized::Bool`
- `lambda::ComplexF64`
"""
function check_heigenpair(A::Array{T, N}, x::Vector{Complex{Float64}}) where {T, N}
    n = size(A, 1)
    m = ndims(A)
    lambda = NaN + NaN*im 
    is_eigenvector = false

    # --- Condition 1: Ax^(m-1) = λx^[m-1] ---
    # We need to compute y = Ax^(m-1)
    # y_i = sum(A[i, j2, ..., jm] * x[j2] * ... * x[jm])
    
    y = zeros(Complex{Float64}, n)
    
    # We can use CartesianIndices to iterate over the input indices
    # We need to iterate j2...jm
    indices_iterator = CartesianIndices(ntuple(_ -> n, m - 1))
    
    for i in 1:n
        sum_val = 0.0 + 0.0im
        for idx_tuple in indices_iterator
            # Construct the full index for A: (i, j2, ..., jm)
            full_idx = (i, idx_tuple.I...)
            
            # Compute product of x components: x[j2]*...*x[jm]
            x_prod = prod(x[k] for k in idx_tuple.I)
            
            sum_val += A[full_idx...] * x_prod
        end
        y[i] = sum_val
    end

    # Now check y_i = λ * x_i^(m-1)
    # λ = y_i / x_i^(m-1)
    
    x_pow = x .^ (m - 1)
    candidate_lambdas = ComplexF64[]
    initial_check_passed = true
    
    for i in 1:n
        # If x[i] is non-zero, calculate a candidate lambda
        if abs(x_pow[i]) > 1e-8 # Using a small tolerance for zero
            push!(candidate_lambdas, y[i] / x_pow[i])
        # If x[i] is zero, y[i] must also be zero for the equation to hold
        elseif abs(y[i]) > 1e-8
            initial_check_passed = false
            # If y[i] is not zero but x[i] is zero, it's impossible (unless λ is inf?)
            # This is a failure.
        end
    end

    if !initial_check_passed || isempty(candidate_lambdas)
        return false, false, NaN + NaN*im
    end

    # Check if all candidate lambdas are consistent (approx. equal)
    first_lambda = candidate_lambdas[1]
    all_consistent = all(isapprox(l, first_lambda, atol=1e-6) for l in candidate_lambdas)
    
    if all_consistent
        is_eigenvector = true
        lambda = first_lambda
    end

    # --- Condition 2: Σxᵢ² = 1 (Normalization) ---
    norm_sum_sq = sum(x.^2)
    is_normalized = isapprox(norm_sum_sq, 1.0, atol=1e-6)

    return is_eigenvector, is_normalized, lambda
end
