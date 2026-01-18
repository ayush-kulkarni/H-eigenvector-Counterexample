using LinearAlgebra
using SymPy

"""
    hyperdeterminant(A::Array{T, 3}) -> T

Computes Cayley's hyperdeterminant for a 2x2x2 tensor A.
"""
function hyperdeterminant(A::Array{T, 3}) where T
    # Using 1-based indexing
    a000, a001 = A[1,1,1], A[1,1,2]
    a010, a011 = A[1,2,1], A[1,2,2]
    a100, a101 = A[2,1,1], A[2,1,2]
    a110, a111 = A[2,2,1], A[2,2,2]

    # Cayley's Hyperdeterminant Formula
    term1 = (a000^2 * a111^2) + (a001^2 * a110^2) + (a010^2 * a101^2) + (a100^2 * a011^2)
    term2 = -2 * (
        (a000 * a001 * a110 * a111) + (a000 * a010 * a101 * a111) +
        (a000 * a100 * a011 * a111) + (a001 * a010 * a101 * a110) +
        (a001 * a100 * a011 * a110) + (a010 * a100 * a011 * a101)
    )
    term3 = 4 * ((a000 * a011 * a101 * a110) + (a001 * a010 * a100 * a111))
    
    return term1 + term2 + term3
end

"""
    identity_tensor_2x2x2() -> Array{Rational{Int}, 3}

Returns the generalized identity tensor for the eigenvalue problem (diagonal 1s).
"""
function identity_tensor_2x2x2()
    I_tens = zeros(Rational{Int}, 2, 2, 2)
    I_tens[1, 1, 1] = 1//1
    I_tens[2, 2, 2] = 1//1
    return I_tens
end

"""
    get_characteristic_coefficients(A::Array{Rational{Int}, 3}) -> Vector{Rational{Int}}

Computes the coefficients of the characteristic polynomial P(λ) = Det(A - λI) using interpolation.
For tensors, this 'Det' is the hyperdeterminant.
"""
function get_characteristic_coefficients(A::Array{Rational{Int}, 3})
    I_tens = identity_tensor_2x2x2()
    
    # Interpolation points (Rational Integers)
    x_points = [Rational{Int}(x) for x in -2:2]
    y_points = Rational{Int}[]

    # Evaluate Det(A - λI) at the 5 points
    for x in x_points
        A_shifted = A .- (x .* I_tens)
        push!(y_points, hyperdeterminant(A_shifted))
    end

    # Solve Vandermonde system V * coeffs = y
    V = [x^p for x in x_points, p in 0:4]
    coeffs = V \ y_points
    
    return coeffs
end

"""
    solve_polynomial_roots(coeffs) -> Vector{Complex}

Finds the roots of the polynomial defined by `coeffs` using SymPy.
"""
function solve_polynomial_roots(coeffs)
    @syms λ
    poly_expr = 0
    for (i, c) in enumerate(coeffs)
        poly_expr += c * λ^(i-1)
    end
    
    # Use SymPy to solve
    roots = SymPy.solve(Eq(poly_expr, 0), λ)
    
    # Convert to complex numbers for display/usage
    return [complex(N(r)) for r in roots]
end

"""
    print_polynomial(coeffs, label)

Helper to print the polynomial in a readable format.
"""
function print_polynomial(coeffs, label)
    terms = String[]
    degrees = ["", "λ", "λ^2", "λ^3", "λ^4"]
    
    for (i, c) in enumerate(coeffs)
        if c != 0
            num = numerator(c)
            den = denominator(c)
            
            sign_str = num < 0 ? " - " : " + "
            abs_num = abs(num)
            val_str = (den == 1) ? "$abs_num" : "$abs_num/$den"
            
            if isempty(terms)
                push!(terms, (num < 0 ? "-" : "") * "$val_str$(degrees[i])")
            else
                push!(terms, "$sign_str$val_str$(degrees[i])")
            end
        end
    end
    
    println("$label Characteristic Equation: P(λ) = " * join(terms) * " = 0")
end

"""
    run_analytical_workflow(tensor, name)

Runs the full analytical workflow: compute poly, print it, find roots.
"""
function run_analytical_workflow(tensor::Array{Rational{Int}, 3}, name::String)
    println("\n--- Analytical Analysis for $name ---")
    coeffs = get_characteristic_coefficients(tensor)
    print_polynomial(coeffs, name)
    
    println("Finding roots analytically (via SymPy)...")
    roots = solve_polynomial_roots(coeffs)
    
    println("Roots (Eigenvalues):")
    # Sort by magnitude for easier reading
    sort!(roots, by=abs, rev=true)
    for r in roots
        println("  λ = $(round(r, digits=6))  (|λ| = $(round(abs(r), digits=6)))")
    end
    return roots
end