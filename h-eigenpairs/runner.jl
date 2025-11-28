using HomotopyContinuation

try
    include("h-eigenpair_eqs_generator.jl")
    include("h-eigenpair_sys_solver.jl")
catch e
    println("Error: Make sure dependent files ('h-eigenpair_eqs_generator.jl', 'h-eigenpair_sys_solver.jl') are in the same directory.")
    rethrow(e)
end

function run_heigenpair_workflow(A, k)
    
    # --- File Output Configuration ---
    equations_output_filename = "./h-eigenpair_eqs_$k.txt"
    solutions_output_filename = "./solutions_$k.txt"

    # --- Main Execution Logic ---
    try
        order_m = ndims(A)
        dim_n = size(A, 1)
        # --- Generate Symbolic Equations ---
        # println("Generating symbolic equations...")
        F, eqs, x_vars, lambda_var = generate_heigenpair_system(A)
        all_vars = [x_vars..., lambda_var]
        # println("Symbolic system generated.")

        # # --- Write Equations to File ---
        # println("\nWriting equations to '$equations_output_filename'...")
        # open(equations_output_filename, "w") do file
        #     write(file, "System of Equations for H-Eigenpairs\n")
        #     write(file, "========================================\n\n")
        #     write(file, "Input Tensor Details:\n  - Order (m): $order_m\n  - Dimension (n): $dim_n\n\n")
        #     write(file, "Variables:\n  - Eigenvalue: $lambda_var\n  - Eigenvector components: $(join([string(v) for v in x_vars], ", "))\n\n")
        #     write(file, "Equations:\n")
        #     for (i, eq) in enumerate(eqs)
        #         write(file, "  Eq. $i:  $(string(eq))\n")
        #     end
        #     write(file, "\n========================================\n")
        # end
        # println("File written successfully.")

        # --- Solve the System and Write Solutions ---
        # println("\nSolving and writing solutions...")
        return solve_and_write_solutions(F, solutions_output_filename)

    catch e
        println("\nAn error occurred during execution:")
        showerror(stdout, e)
        println()
    end
end