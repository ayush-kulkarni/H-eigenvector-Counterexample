using REPL.TerminalMenus

include("analytical/characteristic_polynomial.jl")
include("numerical/numerical_experiment.jl")

function main()
    println("==========================================")
    println("   Tensor H-Eigenvalue Analysis Tool")
    println("==========================================")
    
    options = [
        "Numerical Analysis (HomotopyContinuation)",
        "Analytical Analysis (Characteristic Polynomial & Roots)",
        "Verify Specific Solutions (Analytic Derivation)",
        "Exit"
    ]
    
    menu = RadioMenu(options, pagesize=4)
    choice = request("Choose a workflow:", menu)

    if choice == 1
        println("\n>>> Starting Numerical Analysis...")
        run_numerical_experiment()
        
    elseif choice == 2
        println("\n>>> Starting Analytical Analysis...")
        # Get tensors
        tensor_A, tensor_B = get_example_tensors()
        
        # Convert to rational for exact arithmetic
        tensor_A_rat = rationalize.(Int, tensor_A)
        tensor_B_rat = rationalize.(Int, tensor_B)
        
        run_analytical_workflow(tensor_A_rat, "Tensor A")
        run_analytical_workflow(tensor_B_rat, "Tensor B")
        
    elseif choice == 3
        println("\n>>> Verifying Solutions Analytically...")
        # We run the script as a separate process to maintain a clean environment for SymPy
        run(`julia --project=. analytical/h_eigenvector_verifier.jl`)
        
    else
        println("Exiting.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
