using LinearAlgebra

# Include shared utilities
include("../common/tensor_utils.jl")

"""
    compute_h_eigen_residual(A, x, λ)

Computes the residual vector r = A * x^(m-1) - λ * x^[m-1].
Returns the norm of r.
"""
function compute_h_eigen_residual(A, x, λ)
    m = ndims(A)
    n = size(A, 1)
    
    # Check dimensions
    if length(x) != n
        error("Vector x dimension $(length(x)) does not match tensor dimension $n")
    end
    
    # Compute LHS = A * x^(m-1)
    lhs = zeros(ComplexF64, n)
    
    # Iterate over all indices
    # We want to sum over j2...jm
    # A[i, j2, ..., jm] * x[j2] * ... * x[jm]
    
    # Total indices: m. Iterating over all is n^m.
    # For n=2, m=4, that's 16 iterations. Very fast.
    for idx in CartesianIndices(A)
        # idx is (i, j2, ..., jm)
        val = A[idx]
        
        # Product of x terms corresponding to indices 2 to m
        x_prod = one(ComplexF64)
        for k in 2:m
            x_prod *= x[idx[k]]
        end
        
        # Add to LHS[i]
        lhs[idx[1]] += val * x_prod
    end
    
    # Compute RHS = λ * x^[m-1] (component-wise power)
    rhs = λ .* (x .^ (m - 1))
    
    # Residual
    res_vec = lhs .- rhs
    return norm(res_vec)
end

"""
    verify_solutions_generic(tensor, tensor_name, filename)

Verifies the solutions in `filename` against the tensor `tensor` using the H-eigenvalue equation.
"""
function verify_solutions_generic(tensor, tensor_name, filename)
    println("\n==========================================")
    println("      Verifying $tensor_name       ")
    println("      (Order: $(ndims(tensor)), Dim: $(size(tensor,1)))")
    println("==========================================")
    
    sols = parse_solutions(filename)
    if isempty(sols)
        println("No solutions found in $filename to verify.")
        return
    end
    
    match_count = 0
    total = length(sols)
    tolerance = 1e-6 # Tolerance for residual
    
    for (i, sol) in enumerate(sols)
        λ_val = sol.λ
        vec_val = sol.vec
        
        residual = compute_h_eigen_residual(tensor, vec_val, λ_val)
        
        status = residual < tolerance ? "PASS" : "FAIL"
        if status == "PASS"
            match_count += 1
        end
        
        println("  Sol $i: λ=$(round(λ_val, digits=4))")
        println("         x=$(round.(vec_val, digits=4))")
        println("         Residual: $residual -> $status")
    end
    
    println("\n  Summary: $match_count / $total solutions satisfied the H-eigenpair equation (tol=$tolerance).")
end

function main()
    # Define Tensors
    tensor_A, tensor_B = get_example_tensors()

    # The get_example_tensors might return different tensors based on version.
    # We verify whatever is returned.
    
    # Check if solution files exist
    if !isfile("solutions_A.txt") && !isfile("solutions_B.txt")
        println("No solution files (solutions_A.txt, solutions_B.txt) found.")
        println("Please run the 'Numerical Analysis' workflow first to generate solutions.")
        return
    end

    if isfile("solutions_A.txt")
        verify_solutions_generic(tensor_A, "Tensor A", "solutions_A.txt")
    end
    
    if isfile("solutions_B.txt")
        verify_solutions_generic(tensor_B, "Tensor B", "solutions_B.txt")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end