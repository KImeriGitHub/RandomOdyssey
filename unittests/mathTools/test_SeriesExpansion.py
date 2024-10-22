import numpy as np

from src.mathTools.SeriesExpansion import SeriesExpansion

def test_getFourierConst():
    """
    Test the getFourierConst function by comparing computed Fourier coefficients
    to known coefficients of a test function.
    """
    # Parameters
    M = 1  # Number of functions (can be increased to test vectorization)
    N = 64  # Number of sample points, must be even

    # Known Fourier coefficients
    a0 = 1.0
    a1 = 0.5
    a2 = 0.25
    b1 = -0.3
    b2 = 0.15

    # Generate sample points
    t = np.linspace(-np.pi, np.pi, N+1)[:-1]  # Exclude the last point to get N points

    # Create the test function f(t) with known Fourier coefficients
    f_t = (a0 +
           a1 * np.cos(t) +
           a2 * np.cos(2 * t) +
           b1 * np.sin(t) +
           b2 * np.sin(2 * t))

    # Reshape f_t to match the expected input shape (M x N)
    f_t = f_t.reshape(M, N)

    # Call the getFourierConst function
    CosConst, SinConst = SeriesExpansion.getFourierConst(f_t)

    # Extract the computed coefficients
    computed_a0 = CosConst[0, 0].real
    computed_a1 = CosConst[0, 1].real
    computed_a2 = CosConst[0, 2].real
    computed_b1 = SinConst[0, 1].real
    computed_b2 = SinConst[0, 2].real

    # Print the known and computed coefficients
    print("Known coefficients:")
    print(f"a0 = {a0}, a1 = {a1}, a2 = {a2}, b1 = {b1}, b2 = {b2}")
    print("\nComputed coefficients:")
    print(f"a0 = {computed_a0}, a1 = {computed_a1}, a2 = {computed_a2}, b1 = {computed_b1}, b2 = {computed_b2}")

    # Check if the computed coefficients are close to the known ones
    tol = 1e-6  # Tolerance for floating-point comparison
    assert np.allclose(computed_a0, a0, atol=tol), "a0 does not match"
    assert np.allclose(computed_a1, a1, atol=tol), "a1 does not match"
    assert np.allclose(computed_a2, a2, atol=tol), "a2 does not match"
    assert np.allclose(computed_b1, b1, atol=tol), "b1 does not match"
    assert np.allclose(computed_b2, b2, atol=tol), "b2 does not match"

    print("\nAll coefficients match within the specified tolerance.")
    print("Test passed!")


import numpy as np

def test_getFourierConst_multi():
    """
    Test the getFourierConst function with multiple functions (M > 1)
    by comparing computed Fourier coefficients to known coefficients.
    """
    # Parameters
    M = 3   # Number of functions
    N = 64  # Number of sample points, must be even

    # Generate sample points
    t = np.linspace(-np.pi, np.pi, N+1)[:-1]  # Exclude the last point to get N points

    # Initialize arrays to hold the test functions and known coefficients
    f_t = np.zeros((M, N))
    known_a = np.zeros((M, N//2))
    known_b = np.zeros((M, N//2))

    # Define known Fourier coefficients for each function
    # Function 1
    known_a[0, 0] = 1.0    # a0
    known_a[0, 1] = 0.5    # a1
    known_a[0, 2] = 0.25   # a2
    known_b[0, 1] = -0.3   # b1
    known_b[0, 2] = 0.15   # b2

    # Function 2
    known_a[1, 0] = 0.8
    known_a[1, 1] = -0.4
    known_a[1, 3] = 0.6
    known_b[1, 2] = 0.2
    known_b[1, 4] = -0.1

    # Function 3
    known_a[2, 0] = -0.5
    known_a[2, 2] = 0.3
    known_b[2, 1] = 0.7
    known_b[2, 3] = -0.2
    known_b[2, 5] = 0.05

    # Create the test functions using the known coefficients
    for m in range(M):
        # Initialize the function for current m
        f = np.zeros(N)
        # Add cosine terms
        for n in range(N//2):
            if known_a[m, n] != 0:
                f += known_a[m, n] * np.cos(n * t)
        # Add sine terms
        for n in range(1, N//2):  # Start from n=1 for sine terms
            if known_b[m, n] != 0:
                f += known_b[m, n] * np.sin(n * t)
        # Store the function
        f_t[m, :] = f

    # Call the getFourierConst function
    CosConst, SinConst = SeriesExpansion.getFourierConst(f_t)

    # Extract the computed coefficients
    computed_a = CosConst.real
    computed_b = SinConst.real

    # Tolerance for floating-point comparison
    tol = 1e-6

    # Compare the computed coefficients to the known coefficients
    for m in range(M):
        print(f"\nFunction {m+1}:")
        print("Known a coefficients:", known_a[m, :5])
        print("Computed a coefficients:", computed_a[m, :5])
        print("Known b coefficients:", known_b[m, :5])
        print("Computed b coefficients:", computed_b[m, :5])

        # Assertions for cosine coefficients
        assert np.allclose(computed_a[m, :], known_a[m, :], atol=tol), f"a coefficients do not match for function {m+1}"
        # Assertions for sine coefficients
        assert np.allclose(computed_b[m, :], known_b[m, :], atol=tol), f"b coefficients do not match for function {m+1}"

    print("\nAll coefficients match within the specified tolerance for all functions.")
    print("Test passed!")
