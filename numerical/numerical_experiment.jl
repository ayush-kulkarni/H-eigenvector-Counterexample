using Combinatorics
using LinearAlgebra

# Include dependencies
include("../common/tensor_utils.jl")

# Include the workflow runner
include("tensor_workflow.jl")

"""
    run_check_on_tensors(tensor_B, tensor_A)

Runs the verification workflow on two tensors A and B, and their Kronecker product C = B ⊗ A.
Checks if |λ(C)| ≈ |λ(A)| * |λ(B)|.
"""
function run_check_on_tensors(tensor_B, tensor_A)
    kronecker_product_result = kronecker_product(tensor_B, tensor_A)
    
    println("\nRunning workflow for Tensor A...")
    largest_magnitude_lambda_A = run_heigenpair_workflow(tensor_A, "A")
    
    println("\nRunning workflow for Tensor B...")
    largest_magnitude_lambda_B = run_heigenpair_workflow(tensor_B, "B")
    
    println("\nRunning workflow for Tensor C (B ⊗ A)...")
    largest_magnitude_lambda_C = run_heigenpair_workflow(kronecker_product_result, "C")

    if largest_magnitude_lambda_A == -1.0 || largest_magnitude_lambda_B == -1.0 || largest_magnitude_lambda_C == -1.0
        println("One of the solvers failed to find real solutions (or failed generally). Check logs.")
        return
    end

    product_of_lambdas = largest_magnitude_lambda_A * largest_magnitude_lambda_B

    println("\n--- Results ---")
    println("|λ_A|: $largest_magnitude_lambda_A")
    println("|λ_B|: $largest_magnitude_lambda_B")
    println("|λ_A| * |λ_B|: $product_of_lambdas")
    println("|λ_C|: $largest_magnitude_lambda_C")
    println("Difference: $(abs(product_of_lambdas - largest_magnitude_lambda_C))")

    if !isapprox(product_of_lambdas, largest_magnitude_lambda_C, atol=1e-8)
        println(">>> Check FAILED. The multiplicative property does not hold.")
        println("The following tensors caused the failure:")
        # println("\nTensor A:")
        # display(tensor_A)
        # println("\nTensor B:")
        # display(tensor_B)
    else
        println(">>> Check PASSED. The multiplicative property holds.")
    end
end

"""
    run_numerical_experiment()

Main entry point for the numerical experiment.
"""
function run_numerical_experiment()
    # 1. Run check on known counterexample tensors
    println("Running check on specific example tensors...")
    tensor_A, tensor_B = get_example_tensors()
    run_check_on_tensors(tensor_B, tensor_A)
end

# Ensure it doesn't run automatically if included
# main()