# Tensor Kronecker Product H-Eigenvector Counterexamples

Computational companion code for the paper **"Dominant H-Eigenvectors of Tensor Kronecker Products Do Not Decouple"** (Ayush Kulkarni, Prof. David Gleich — Purdue University), currently in review at *Linear Algebra and its Applications*. [arXiv:2508.19902](https://arxiv.org/abs/2508.19902)

## Background: what this project is actually about

**Tensors** are the higher-dimensional generalization of matrices. A matrix is a 2D grid of numbers; an order-3 tensor is a 3D "cube" of numbers, order-4 is a 4D array, and so on.

Tensors have an analogous concept called **H-eigenvalues** (introduced by Qi, 2005), defined by the equation
```
A x^(m-1) = λ x^(m-1),      subject to  Σ xᵢ² = 1
```

where `A` is an order-`m` tensor. Unlike the matrix case, this is a *nonlinear* system of polynomial equations, so finding H-eigenpairs generally requires either numerical polynomial solvers or exact algebraic methods. Both are implemented in this repo.

**The Kronecker product** is a standard way to combine two matrices or tensors into a larger one (used throughout linear algebra, network science, and quantum information). For matrices, there's a clean, well-known rule: **the eigenvalues of a Kronecker product are just the products of the two factors' eigenvalues.** This "decoupling" property is extremely useful in practice. It means you can compute the spectral behavior of a huge combined system just from the two small pieces, without ever building the large matrix.

**The open question this paper answers:** does that same decoupling property hold for tensors' *dominant* (largest-magnitude) H-eigenvectors? If tensor `A` has dominant H-eigenpair `(λ_A, x_A)` and tensor `B` has `(λ_B, x_B)`, is the dominant H-eigenpair of their Kronecker product `B ⊗ A` simply `(λ_A·λ_B, x_A ⊗ x_B)`?

**This paper's result: no, in general it does not.** It constructs explicit tensor pairs where the multiplicative property `|λ(B⊗A)| = |λ(A)|·|λ(B)|` fails, disproving a natural extension of the matrix case. This matters for anyone modeling systems as tensor products (e.g., multi-way data analysis, higher-order network models). You cannot assume the convenient "solve the small pieces, combine analytically" shortcut carries over from matrices to tensors.

## Methodology: two independent verification pipelines

To trust a candidate counterexample, every result is cross-checked by two independently implemented solvers:

1. **Numerical (`numerical/`)** — `h_eigenpair_generator.jl` symbolically builds the polynomial system above for a given tensor; `h_eigenpair_solver.jl` hands that system to [`HomotopyContinuation.jl`](https://www.juliahomotopycontinuation.org/) (a Julia package for finding *all* complex solutions of a polynomial system via numerical homotopy continuation), then extracts the largest-magnitude real eigenvalue. `h_eigenpair_verifier.jl` independently re-checks a candidate `(λ, x)` pair against the raw tensor equation as a sanity check.

2. **Analytical (`analytical/`)** — for dimension-2 tensors, `characteristic_polynomial.jl` computes the *exact* characteristic polynomial using a Sylvester-matrix resultant (with exact `Rational{Int}` arithmetic, avoiding any floating-point error in the coefficients), then extracts its roots via companion-matrix eigendecomposition. This provides an algebraically independent check on the numerical solver's output.

3. **Comparison** — `numerical_experiment.jl` computes `|λ_A|`, `|λ_B|`, and `|λ_C|` for `C = B ⊗ A`, and checks whether `|λ_A|·|λ_B| ≈ |λ_C|`. A mismatch is a counterexample to the decoupling conjecture.

A separate exploratory script, `quasi-diagonality-checker.py`, tests a related structural conjecture (whether a "quasi-diagonal" tensor structure is preserved under Kronecker products) using NumPy.

## Codebase layout

```
main.jl                              Interactive CLI (menu-driven entry point)
common/
  tensor_utils.jl                    Kronecker product, symmetric tensor generation,
                                      and the paper's hand-constructed example tensors
numerical/
  h_eigenpair_generator.jl           Builds the symbolic H-eigenpair polynomial system
  h_eigenpair_solver.jl              Solves it via HomotopyContinuation.jl
  h_eigenpair_verifier.jl            Independently re-verifies a candidate (λ, x) pair
  numerical_experiment.jl            Orchestrates the A / B / C=B⊗A comparison
  tensor_workflow.jl                 Glue code tying generator → solver together
analytical/
  characteristic_polynomial.jl       Exact resultant + companion-matrix eigensolver
  h_eigenvector_verifier.jl          Analytical-side verification
quasi-diagonality-checker.py         Related structural conjecture, NumPy
benchmarks/
  pipeline_benchmark.jl              Standalone performance comparison (see below);
                                      does not affect the main pipeline
solutions_A.txt, solutions_B.txt,
solutions_C.txt                      Solver output logs for the paper's example tensors
```

## Running it

Requires [Julia](https://julialang.org/) (developed against 1.12) and the packages pinned in `Project.toml`/`Manifest.toml`.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # one-time dependency install
julia --project=. main.jl                              # interactive menu
```

The menu lets you: (1) run the check on the paper's known counterexample tensors, (2) run it on freshly generated random symmetric tensors, (3) run the analytical (exact) workflow, or (4) verify a specific solution analytically.

## Performance analysis: how the benchmark numbers were calculated

The pipeline's `kronecker_product` function currently builds the combined tensor by first creating two full-size temporary copies of the inputs (each one stretched out to match the final size), then multiplying those two copies together to get the answer. That works, but it means extra time and memory go into building copies that are thrown away immediately after.

To see how much that actually costs, a separate comparison script (`benchmarks/pipeline_benchmark.jl`) computes the exact same result a different way: instead of building those two temporary copies, it fills in every cell of the final answer directly, one at a time, with no throwaway copies at all. The same kind of comparison was run for `generate_symmetric_tensor`, which currently builds its output using a general-purpose combinatorics library rather than a simple direct pass over the tensor.

Both versions of each function were run hundreds of times on the same, real tensors (the sizes actually used elsewhere in this project) to get a stable reading, and their speed and memory use were compared. A quick check also confirmed both versions produce identical output before anything was timed, so the comparison is apples-to-apples.

Measured results (reproduce with `julia --project=. benchmarks/pipeline_benchmark.jl`):

| Function | Current implementation | Alternative | Runtime | Memory |
|---|---|---|---|---|
| `kronecker_product` | builds two temporary copies, then multiplies | fills the answer directly, no copies | **25.1% faster** | **68.4% less** |
| `generate_symmetric_tensor` | general combinatorics library | direct single pass | **83.9% faster** | **86.6% less** |

These numbers describe a comparison used to measure the idea, not a change merged into the shipped pipeline — `common/tensor_utils.jl` itself is unchanged.

### A note on counterexample-discovery metrics

An earlier draft considered citing a count like "N counterexamples surfaced across M random trials." That framing was checked empirically and dropped: running `run_check_on_random_tensors` for 105 uniformly-random order-3, dimension-2 trials found **zero** violations of the decoupling property. The paper's actual counterexamples (see `get_example_tensors()` in `common/tensor_utils.jl`) are specific, hand-derived tensors, not something uniform random sampling turns up at any practical trial count — consistent with the paper itself being a targeted analytical construction backed by computational verification, rather than a brute-force search.
