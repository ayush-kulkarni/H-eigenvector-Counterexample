# import Pkg; 
# Pkg.add("Combinatorics")
# Pkg.add("HomotopyContinuation")
# Pkg.add("LinearAlgebra")

using Combinatorics

try
    include("h-eigenpairs/runner.jl")

catch e
    println("Error: Make sure 'runner.jl' is in the same directory as this script.")
    rethrow(e)
end

function kronecker_product(A::AbstractArray, B::AbstractArray)
    # Get the sizes of the input tensors
    sA = size(A)
    sB = size(B)

    # Ensure the tensors have the same number of dimensions
    if ndims(A) != ndims(B)
        error("Input tensors must have the same number of dimensions. Got $(ndims(A)) and $(ndims(B))")
    end

    # The Kronecker product can be computed efficiently by repeating each element of A to match the size of B, and tiling B to match the size of A followed by an element-wise product.
    A_repeated = repeat(A, inner=sB)
    B_repeated = repeat(B, outer=sA)

    return A_repeated .* B_repeated
end

function run_check_on_tensors(tensor_B, tensor_A)
    kronecker_product_result = kronecker_product(tensor_B, tensor_A)
    largest_magnitude_lambda_A = run_heigenpair_workflow(tensor_A, "A")
    largest_magnitude_lambda_B = run_heigenpair_workflow(tensor_B, "B")
    largest_magnitude_lambda_C = run_heigenpair_workflow(kronecker_product_result, "C")

    product_of_lambdas = largest_magnitude_lambda_A * largest_magnitude_lambda_B

    if !isapprox(product_of_lambdas, largest_magnitude_lambda_C, atol=1e-8)
        println("Check FAILED. Terminating program. ")
        println("|λ_A| * |λ_B|: $product_of_lambdas")
        println("Largest |λ_C|: $largest_magnitude_lambda_C")
        println("Difference: $(abs(product_of_lambdas - largest_magnitude_lambda_C))")
        println("The following tensors caused the failure:")
        println("\nTensor A:")
        display(tensor_A)
        println("\nTensor B:")
        display(tensor_B)
        println("\nKronecker Product (B ⊗ A):")
        display(kronecker_product_result)
        return # Terminate after the first failure
    else
        println("|λ_A| * |λ_B|: $product_of_lambdas")
        println("Largest |λ_C|: $largest_magnitude_lambda_C")
        println("Difference: $(abs(product_of_lambdas - largest_magnitude_lambda_C))")
    end
end

function run_check_on_random_tensors(num_tests)
    for i in 1:num_tests
        println("Test $i")
        # Generate random symmetric tensors A and B
        tensor_A = generate_symmetric_tensor(6, (2, 2, 2, 2, 2, 2))
        tensor_B = generate_symmetric_tensor(6, (2, 2, 2, 2, 2, 2))

        # Compute their Kronecker product
        kronecker_product_result = kronecker_product(tensor_B, tensor_A)

        # Run the workflow and compare results
        largest_magnitude_lambda_A = run_heigenpair_workflow(tensor_A, "A")
        if (largest_magnitude_lambda_A == -1)
            # println("Solver failed for tensor A. Skipping this test.")
            continue
        end
        largest_magnitude_lambda_B = run_heigenpair_workflow(tensor_B, "B")
        if (largest_magnitude_lambda_B == -1)
            # println("Solver failed for tensor B. Skipping this test.")
            continue
        end
        largest_magnitude_lambda_C = run_heigenpair_workflow(kronecker_product_result, "C")
        if (largest_magnitude_lambda_C == -1)
            # println("Solver failed for Kronecker product tensor C. Skipping this test.")
            continue
        end

        product_of_lambdas = largest_magnitude_lambda_A * largest_magnitude_lambda_B

        if !isapprox(product_of_lambdas, largest_magnitude_lambda_C, atol=1e-8)
            println("Check FAILED. Terminating program. ")
            println("|λ_A| * |λ_B|: $product_of_lambdas")
            println("Largest |λ_C|: $largest_magnitude_lambda_C")
            println("Difference: $(abs(product_of_lambdas - largest_magnitude_lambda_C))")
            println("The following tensors caused the failure:")
            println("\nTensor A:")
            display(tensor_A)
            println("\nTensor B:")
            display(tensor_B)
            println("\nKronecker Product (B ⊗ A):")
            display(kronecker_product_result)
            return # Terminate after the first failure
        end
    end
    # If the loop completes, all checks passed and the program exits silently.
end

function generate_symmetric_tensor(dim, size)
    tensor = zeros(size)
    n = size[1]
    # This implementation assumes all dimensions have the same size
    if !all(s -> s == n, size)
        error("generate_symmetric_tensor requires all dimensions to have the same size.")
    end

    # Iterate through canonical indices (where i_1 <= i_2 <= ... <= i_d)
    for idx_tuple in Combinatorics.with_replacement_combinations(1:n, dim)
        val = rand(0.0:0.1:1.0)
        for p_tuple in unique(Combinatorics.permutations(idx_tuple))
            tensor[p_tuple...] = val
        end
    end
    return tensor
end


# -- Randomly test tensors to see if we can find one that doesn't work -- 
run_check_on_random_tensors(5)


# Run Hyperdeterminant Analysis on the provided tensor
# try
#     # Load dependencies and data
#     include("hyperdeterminant/run-hyperdeterminant.jl")
#     include("hyperdeterminant/hyperdeterminant.jl")
    
#     # Call the main function with the loaded tensor
#     # analyze_hyperdeterminant(A)

# catch e
#     println("\nAn error occurred during script execution:")
#     showerror(stdout, e)
#     println()
# end