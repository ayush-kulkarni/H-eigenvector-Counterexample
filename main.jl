using REPL.TerminalMenus

include("analytical/characteristic_polynomial.jl")
include("numerical/numerical_experiment.jl")

function main()
    println("==========================================")
    println("   Tensor H-Eigenvalue Analysis Tool")
    println("==========================================")
    
    options = [
        "Numerical Analysis: Known Counterexample",
        "Numerical Analysis: Random Tensors",
        "Analytical Analysis (Characteristic Polynomial & Roots)",
        "Verify Specific Solutions (Analytic Derivation)",
        "Exit"
    ]
    
    menu = RadioMenu(options, pagesize=5)
    choice = request("Choose a workflow:", menu)

    if choice == 1
        println("\n>>> Starting Numerical Analysis (Known Counterexample)...")
        run_numerical_experiment()
        
    elseif choice == 2
        println("\n>>> Starting Numerical Analysis (Random Tensors)...")
        
        print("Enter number of tests (default 1): ")
        input_tests = readline()
        num_tests = isempty(strip(input_tests)) ? 1 : parse(Int, strip(input_tests))
        
        print("Enter tensor order (default 3): ")
        input_order = readline()
        order = isempty(strip(input_order)) ? 3 : parse(Int, strip(input_order))
        
        print("Enter tensor dimension (default 2): ")
        input_dim = readline()
        dimension = isempty(strip(input_dim)) ? 2 : parse(Int, strip(input_dim))
        
        run_check_on_random_tensors(num_tests; order=order, dimension=dimension)

    elseif choice == 3
        println("\n>>> Starting Analytical Analysis...")
        # Get tensors
        tensor_A, tensor_B = get_example_tensors()
        
        # Convert to rational for exact arithmetic
        tensor_A_rat = rationalize.(Int, tensor_A)
        tensor_B_rat = rationalize.(Int, tensor_B)
        
        run_analytical_workflow(tensor_A_rat, "Tensor A")
        run_analytical_workflow(tensor_B_rat, "Tensor B")
        
    elseif choice == 4
        println("\n>>> Verifying Solutions Analytically...")
        run(`julia --project=. analytical/h_eigenvector_verifier.jl`)
        
    else
        println("Exiting.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
