using LinearAlgebra
using Symbolics

"""
    get_polynomial_coefficients(A::Array{T, N}, row_idx::Int)

Extracts coefficients of the homogeneous polynomial defined by the `row_idx`-th slice of tensor A.
Assumes dimension size is 2.
"""
function get_polynomial_coefficients(A::Array{T, N}, row_idx::Int) where {T, N}
    # Degree of polynomial is order - 1
    d = N - 1
    # We want coefficients for terms: x^d, x^{d-1}y, ..., y^d
    coeffs = zeros(T, d + 1)
    
    # Iterate over all indices for the remaining dimensions (m-1 dimensions)
    # Each index is in {1, 2}
    iter_dims = ntuple(_ -> 2, d)
    
    for idx in CartesianIndices(iter_dims)
        # Count number of 2s in the index tuple (corresponds to power of y)
        num_twos = count(x -> x == 2, Tuple(idx))
        
        # Access tensor value. Note: First index is row_idx, rest are from idx
        val = A[row_idx, idx]
        
        # Add to the appropriate coefficient (1-based index: 0 twos -> index 1)
        coeffs[num_twos + 1] += val
    end
    
    return coeffs
end

"""
    build_sylvester_matrix(coeffs1, coeffs2)

Constructs the Sylvester matrix for two polynomials defined by `coeffs1` and `coeffs2`.
"""
function build_sylvester_matrix(coeffs1, coeffs2)
    d = length(coeffs1) - 1 # Degree
    size_mat = 2 * d
    M = zeros(eltype(coeffs1), size_mat, size_mat)
    
    # Fill top half with coeffs1
    for i in 1:d
        for j in 1:(d+1)
            M[i, i + j - 1] = coeffs1[j]
        end
    end
    
    # Fill bottom half with coeffs2
    for i in 1:d
        for j in 1:(d+1)
            M[d + i, i + j - 1] = coeffs2[j]
        end
    end
    
    return M
end

"""
    compute_resultant_dim2(A::Array{T, N})

Computes the resultant of the system of polynomials defined by the tensor A (dimension 2).
Uses the Sylvester matrix determinant.
"""
function compute_resultant_dim2(A::Array{T, N}) where {T, N}
    # Get coefficients for the two polynomials (from rows 1 and 2)
    c1 = get_polynomial_coefficients(A, 1)
    c2 = get_polynomial_coefficients(A, 2)
    
    # Build Sylvester matrix
    S = build_sylvester_matrix(c1, c2)
    
    # Compute determinant
    return det(S)
end

"""
    identity_tensor(n::Int, m::Int) -> Array{Rational{Int}, m}

Returns the generalized identity tensor for the eigenvalue problem.
I[i, i, ..., i] = 1, others 0.
"""
function identity_tensor(n::Int, m::Int)
    sz = ntuple(_ -> n, m)
    I_tens = zeros(Rational{Int}, sz)
    for i in 1:n
        idx = ntuple(_ -> i, m)
        I_tens[idx...] = 1//1
    end
    return I_tens
end

"""
    get_characteristic_coefficients(A::Array{Rational{Int}}) -> Vector{Rational{Int}}

Computes the coefficients of the characteristic polynomial P(λ) = Res(A - λI) using interpolation.
Works for any order tensor (dim 2).
"""
function get_characteristic_coefficients(A::Array{Rational{Int}})
    m = ndims(A)
    n = size(A, 1)
    
    if n != 2
        error("Analytical analysis currently only supports dimension n=2.")
    end

    # Degree of Characteristic Polynomial: D = n * (m-1)^(n-1)
    # For n=2: D = 2 * (m-1)
    degree_poly = 2 * (m - 1)
    
    # We need D+1 points for interpolation
    # Use points centered around 0
    start_pt = -(degree_poly ÷ 2)
    x_points = [Rational{Int}(x) for x in start_pt:(start_pt + degree_poly)]
    
    # If not enough points (e.g. degree is small), ensure at least degree+1
    if length(x_points) < degree_poly + 1
        x_points = [Rational{Int}(x) for x in 0:degree_poly]
    end

    I_tens = identity_tensor(n, m)
    y_points = Rational{Int}[]

    # Evaluate Resultant(A - λI) at the points
    for x in x_points
        A_shifted = A .- (x .* I_tens)
        
        # Use general resultant computation
        # Note: Resultant is proportional to the hyperdeterminant/characteristic poly
        push!(y_points, compute_resultant_dim2(A_shifted))
    end

    # Solve Vandermonde system V * coeffs = y
    # V[i, j] = x_points[i]^(j-1)
    V = [x^p for x in x_points, p in 0:degree_poly]
    
    # Solve system using precise rational arithmetic
    coeffs = V \ y_points
    
    return coeffs
end

"""
    solve_polynomial_roots(coeffs) -> Vector{ComplexF64}

Finds the roots of the polynomial defined by `coeffs` using the companion matrix method.
"""
function solve_polynomial_roots(coeffs)
    # P(λ) = c0 + c1λ + ... + cDλ^D
    n = length(coeffs) - 1
    
    # Trim trailing zeros (leading coeffs in high powers)
    while n > 0 && abs(coeffs[n+1]) < 1e-12
        n -= 1
    end
    
    if n == 0 return ComplexF64[] end

    # Monic coefficients
    monic_coeffs = [ComplexF64(c / coeffs[n+1]) for c in coeffs[1:n]]
    
    # Construct Companion Matrix
    C = zeros(ComplexF64, n, n)
    for i in 1:n-1
        C[i+1, i] = 1.0
    end
    for i in 1:n
        C[i, n] = -monic_coeffs[i]
    end
    
    # Eigenvalues of companion matrix are roots of the polynomial
    return eigvals(C)
end

"""
    print_polynomial(coeffs, label)

Helper to print the polynomial in a readable format using Symbolics.
"""
function print_polynomial(coeffs, label)
    @variables λ
    poly_expr = sum(coeffs[i] * λ^(i-1) for i in 1:length(coeffs))
    println("$label Characteristic Equation: P(λ) = $poly_expr = 0")
end

"""
    run_analytical_workflow(tensor, name)

Runs the full analytical workflow: compute poly, print it, find roots.
"""
function run_analytical_workflow(tensor::Array{Rational{Int}}, name::String)
    println("\n--- Analytical Analysis for $name ---")
    println("Tensor Order: $(ndims(tensor)), Dimension: $(size(tensor,1))")
    
    coeffs = get_characteristic_coefficients(tensor)
    print_polynomial(coeffs, name)
    
    println("Finding roots analytically (via Companion Matrix)...")
    roots = solve_polynomial_roots(coeffs)
    
    println("Roots (Eigenvalues):")
    # Sort by magnitude for easier reading
    sort!(roots, by=abs, rev=true)
    for r in roots
        println("  λ = $(round(r, digits=6))  (|λ| = $(round(abs(r), digits=6)))")
    end
    return roots
end
