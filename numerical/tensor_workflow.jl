using HomotopyContinuation

try
    include("h_eigenpair_generator.jl")
    include("h_eigenpair_solver.jl")
catch e
    println("Error: Make sure dependent files ('h_eigenpair_generator.jl', 'h_eigenpair_solver.jl') are in the same directory.")
    rethrow(e)
end

"""
    run_heigenpair_workflow(A::Array, k::String) -> Float64

Orchestrates the H-eigenpair calculation workflow for a given tensor.
1. Generates the system of equations.
2. Solves the system using HomotopyContinuation.
3. Writes results to files.

# Arguments
- `A::Array`: The input tensor.
- `k::String`: A label/identifier for the tensor (used in output filenames).

# Returns
- `largest_magnitude::Float64`: The spectral radius (largest absolute eigenvalue) found.
"""
function run_heigenpair_workflow(A, k)
    
    # --- File Output Configuration ---
    # Output files will be created in the current working directory
    equations_output_filename = "h_eigenpair_eqs_$k.txt"
    solutions_output_filename = "solutions_$k.txt"

    # --- Main Execution Logic ---
    try
        order_m = ndims(A)
        dim_n = size(A, 1)
        
        # --- Generate Symbolic Equations ---
        # println("Generating symbolic equations for tensor $k...")
        F, eqs, x_vars, lambda_var = generate_heigenpair_system(A)
        all_vars = [x_vars..., lambda_var]
        
        # --- Write Equations to File ---
        # (Optional: can be uncommented if debugging is needed)
        # open(equations_output_filename, "w") do file
        #     write(file, "System of Equations for H-Eigenpairs\n")
        #     write(file, "========================================\n")
        #     write(file, "Tensor: $k\n")
        #     # ... (rest of logging)
        # end

        # --- Solve the System and Write Solutions ---
        # println("Solving and writing solutions for tensor $k...")
        return solve_and_write_solutions(F, solutions_output_filename)

    catch e
        println("\nAn error occurred during workflow execution for tensor $k:")
        showerror(stdout, e)
        println()
        return -1.0
    end
end
