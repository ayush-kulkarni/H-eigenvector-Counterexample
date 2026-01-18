using HomotopyContinuation

"""
    solve_and_write_solutions(F::System, solutions_output_filename::String) -> Float64

Solves the polynomial system `F` using HomotopyContinuation.jl and writes the solutions to `solutions_output_filename`.
Returns the largest magnitude of real eigenvalues found.

# Arguments
- `F::System`: The system of equations.
- `solutions_output_filename::String`: Path to save the solutions.

# Returns
- `largest_magnitude::Float64`: The largest absolute value of the eigenvalue λ among all solutions (though the logic seems to track both all and real, returns local max).
"""
function solve_and_write_solutions(F, solutions_output_filename)
    local_largest_magnitude_lambda = 0.0
    # println("Solving the system with HomotopyContinuation.jl...")
    result = HomotopyContinuation.solve(F)
    # println("Solver finished.")

    # println("Writing solutions to '$solutions_output_filename'...")
    all_sols = solutions(result)
    real_sols = real_solutions(result)
    real_sols_count = length(real_sols)

    if (real_sols_count == 0)
        # println("No real solutions found.")
        return -1.0
    end

    open(solutions_output_filename, "w") do file
        write(file, "Solutions for H-Eigenpair System\n")
        write(file, "========================================\n\n")
        write(file, "Found $(length(all_sols)) total solutions (real and complex).\n")
        write(file, "Of these, $real_sols_count are real solutions.\n")
        write(file, "----------------------------------------\n")

        for (i, sol) in enumerate(all_sols)
            write(file, "\n--- Solution $i ---\n")

            λ_val = sol[end]
            x_val = sol[1:end-1]
            # It is better to calculate magnitude from the high-precision value
            magnitude_lambda = abs(λ_val)

            write(file, "  λ = $(round(λ_val, digits=12))\n")
            write(file, "  Magnitude of λ = $(round(magnitude_lambda, digits=12))\n")
            
            if magnitude_lambda > local_largest_magnitude_lambda
                local_largest_magnitude_lambda = magnitude_lambda
            end
            
            write(file, "  x = [\n")
            for val in x_val
                write(file, "        $(round(val, digits=12))\n")
            end
            write(file, "      ]\n")
        end
        write(file, "\n========================================\n")

        # Verify if real max matches global max
        local_largest_magnitude_lambda_2 = 0.0
        for (j, sol_2) in enumerate(real_sols)
            λ_val_2 = sol_2[end]
            magnitude_lambda_2 = abs(λ_val_2)
            if magnitude_lambda_2 > local_largest_magnitude_lambda_2
                local_largest_magnitude_lambda_2 = magnitude_lambda_2
            end
        end

        # This check implies we expect the spectral radius to be attained by a real eigenvalue?
        # Or just checking consistency.
        if (local_largest_magnitude_lambda_2 == local_largest_magnitude_lambda)
            # print("Largest magnitude of real solutions matches the largest magnitude of all solutions.\n")
        else
            # return -1
            # print("WARNING: Largest magnitude of real solutions does not match the largest magnitude of all solutions.\n")
        end
    end
    return local_largest_magnitude_lambda
end
