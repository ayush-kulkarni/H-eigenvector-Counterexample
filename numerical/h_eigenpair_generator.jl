using HomotopyContinuation

"""
    generate_heigenpair_system(A::Array)

Generates the system of polynomial equations defining the H-eigenpairs of a tensor `A`.

The H-eigenvalue equation for an m-th order tensor A is:
    A * x^(m-1) = λ * x^(m-1)
    
Subject to the normalization constraint:
    ∑ x_i^2 = 1

# Arguments
- `A::Array`: The input tensor (any order).

# Returns
- `F::System`: The system of equations for HomotopyContinuation.
- `expressions::Vector`: The symbolic expressions (LHS - RHS).
- `x::Vector{Variable}`: The symbolic variables for the eigenvector.
- `λ::Variable`: The symbolic variable for the eigenvalue.
"""
function generate_heigenpair_system(A::Array)
    # Get the order (m) and dimension (n) of the tensor
    m = ndims(A)
    n = size(A, 1)

    if !all(d == n for d in size(A))
        error("Tensor must have all dimensions equal (supersymmetric shape).")
    end

    @var x[1:n] λ
    all_vars = [x..., λ]

    # Initialize an empty array to store the expressions (lhs - rhs)
    expressions = []

    # --- Generate the first n expressions ---
    # Equation: (A * x^(m-1))_i = λ * x_i^(m-1)
    # Note: The RHS in the original code was λ * x[i]^(m-1).
    # Standard definition usually is λ * x[i]^[m-1] which usually means component-wise power if written that way,
    # but H-eigenvalue definition is A x^(m-1) = λ x^[m-1].
    # Let's verify standard definition:
    # A x^{m-1} is a vector where i-th component is sum_{j_2...j_m} A_{i j_2...j_m} x_{j_2}...x_{j_m}.
    # The RHS is λ * x_i^(m-1) (Qi 2005).
    
    for i in 1:n
        lhs = 0
        indices_iterator = CartesianIndices(ntuple(_ -> n, m - 1))
        for idx_tuple in indices_iterator
            tensor_element = A[i, idx_tuple.I...]
            x_product = prod(x[idx_tuple[k]] for k in 1:(m-1))
            lhs += tensor_element * x_product
        end
        rhs = λ * x[i]^(m-1)
        
        # Create the expression (lhs - rhs) and add it to our list
        push!(expressions, lhs - rhs)
    end

    # --- Generate the normalization expression ---
    # ∑ x_i^2 = 1
    normalization_expr = sum(x[i]^2 for i in 1:n) - 1
    push!(expressions, normalization_expr)

    # --- Create the System for HomotopyContinuation.jl ---
    # The System is built directly from the expressions and variables.
    F = System(expressions, variables = all_vars)

    # Return the system and the expressions for logging purposes.
    return F, expressions, x, λ
end