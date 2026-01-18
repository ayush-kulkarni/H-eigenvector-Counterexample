using SymPy

# Include shared utilities
include("../common/tensor_utils.jl")

"""
    solve_and_verify(tensor, tensor_name, filename)

Derives the symbolic formula for t(λ) for a given tensor and verifies it against numerical solutions.
"""
function solve_and_verify(tensor, tensor_name, filename)
    println("\n==========================================")
    println("      Processing $tensor_name       ")
    println("==========================================")
    
    # 1. Setup Tensor Equations
    A111 = tensor[1, 1, 1]; A121 = tensor[1, 2, 1]
    A112 = tensor[1, 1, 2]; A122 = tensor[1, 2, 2]
    A211 = tensor[2, 1, 1]; A221 = tensor[2, 2, 1]
    A212 = tensor[2, 1, 2]; A222 = tensor[2, 2, 2]

    @syms t λ

    # Equation 1 (i=1): A111 + 2*A121*t + A122*t^2 = λ
    # Rearranged to Quadratic Form: (A122)*t^2 + (2*A121)*t + (A111 - λ) = 0
    a1 = A122
    b1 = 2 * A121
    c1 = A111 - λ
    
    # Equation 2 (i=2): A211 + 2*A212*t + A222*t^2 = λ*t^2
    # Rearranged to Quadratic Form: (A222 - λ)*t^2 + (2*A212)*t + (A211) = 0
    a2 = A222 - λ
    b2 = 2 * A212
    c2 = A211
    
    # 2. Derive Symbolic Formula for t(λ)
    println("Deriving symbolic formula using algebraic elimination...")
    
    # We solve the system of two quadratics by eliminating the t^2 term.
    # Eq 1: a1*t^2 + b1*t + c1 = 0
    # Eq 2: a2*t^2 + b2*t + c2 = 0
    # Multiply Eq 1 by a2 and Eq 2 by a1, then subtract:
    # a2(b1*t + c1) - a1(b2*t + c2) = 0
    # (a2*b1 - a1*b2) * t + (a2*c1 - a1*c2) = 0
    # t = (a1*c2 - a2*c1) / (a2*b1 - a1*b2)
    
    numerator = a1*c2 - a2*c1
    denominator = a2*b1 - a1*b2
    
    if denominator == 0
        # Fallback for degenerate cases where elimination fails (rare for general tensors)
        println("  Denominator is zero. Falling back to direct solve...")
        sols = SymPy.solve(Eq(a1*t^2 + b1*t + c1, 0), t)
        t_formula = simplify(sols[1])
    else
        t_formula = simplify(numerator / denominator)
    end
    
    println("  Formula Derived: t(λ) = $t_formula")

    # 3. Verify against File
    println("\nVerifying against $filename...")
    file_sols = parse_solutions(filename)
    
    match_count = 0
    total = length(file_sols)
    
    # SymPy function for evaluation
    # We substitute λ and evaluate numerically
    
    for (i, sol) in enumerate(file_sols)
        λ_val = sol.λ
        vec_file = sol.vec
        
        # Evaluate formula: t_eval = t_formula(λ_val)
        # Substitute symbolic λ with complex value
        
        # Evaluate using N() to get numerical result
        t_sym_val = N(t_formula.subs(λ, λ_val))
        
        # Convert SymPy number to Julia ComplexF64
        # We can cast via string for robustness or use complex(t_sym_val)
        t_eval = complex(t_sym_val)
        
        # Theoretical Vector [1, t]
        vec_theo = [1.0 + 0.0im, t_eval]
        
        # Check match
        if is_parallel(vec_file, vec_theo)
            match_count += 1
            status = "MATCH"
        else
            # Special check for eigenvectors like [0, 1] which fail the t=x2/x1 model
            if abs(vec_file[1]) < 1e-6 && abs(vec_file[2]) > 1e-6
                status = "SKIP (Vertical Vector [0,1])"
                # This is technically a match for the system, but a fail for the "t" formula
                # So we don't count it as a formula match, but we acknowledge it.
            else
                status = "FAIL"
            end
        end
        
        println("  Sol $i: λ=$(round(λ_val, digits=4)) -> Status: $status")
        if status == "FAIL"
           println("     File: $(round.(vec_file, digits=4))")
           println("     Theo: $(round.(vec_theo, digits=4))")
        end
    end
    
    println("  Summary: $match_count / $total solutions matched the derived formula.")
end

function main()
    # Define Tensors
    tensor_A, tensor_B = get_example_tensors()

    solve_and_verify(tensor_A, "Tensor A", "solutions_A.txt")
    solve_and_verify(tensor_B, "Tensor B", "solutions_B.txt")
end

main()
