import warnings
import numpy as np
from scipy.integrate import quad

# pylint: disable=E0611
from scipy.special import expi
from scipy.optimize import minimize


def Gaussian_exp(norm_s_i, factor):
    return np.exp(-norm_s_i / (2 * factor))


def custom_expi(x):
    return 0 if x == 0 else expi(x)


class LCDSymmEvenOptimizer:
    def __init__(self, b_max, dim, use_integral=False):
        self.b_max = b_max
        self.dim = dim
        self.D1 = self.calc_D1_symm_even()
        self.use_integral_default = use_integral

    @staticmethod
    def integrand_D1_symm_even(b, dim):
        return b * ((b**2) / (1 + b**2)) ** (dim / 2)

    def calc_D1_symm_even(self):
        res, _ = quad(
            LCDSymmEvenOptimizer.integrand_D1_symm_even, 0, self.b_max, args=(self.dim,)
        )
        return res

    @staticmethod
    def integrand_D2_symm_even(b, L, N, s_vectors):
        term1 = 2 * b / (2 * L)
        term2 = ((2 * b**2) / (1 + 2 * b**2)) ** (N / 2)
        term3 = sum(
            Gaussian_exp(np.linalg.norm(s_i) ** 2, (1 + 2 * b**2))
            for s_i in s_vectors
        )
        return term1 * term2 * term3

    @staticmethod
    def integrand_D3_symm_even(b, L, s_vectors):
        term1 = 2 * b / (2 * L) ** 2
        term2 = sum(
            sum(
                Gaussian_exp(
                    np.linalg.norm(s_vectors[i] - s_vectors[j]) ** 2, 2 * b**2
                )
                + Gaussian_exp(
                    np.linalg.norm(s_vectors[i] + s_vectors[j]) ** 2, 2 * b**2
                )
                for j in range(L)
            )
            for i in range(L)
        )
        return term1 * term2

    def calc_D2_symm_even(self, s_vectors):
        L = len(s_vectors)
        N = len(s_vectors[0])
        res, _ = quad(
            LCDSymmEvenOptimizer.integrand_D2_symm_even,
            0,
            self.b_max,
            args=(L, N, s_vectors),
        )
        return res

    def calc_D3_symm_even(self, L, s_vectors, use_integral=False):
        if use_integral:
            res, _ = quad(
                LCDSymmEvenOptimizer.integrand_D3_symm_even,
                0,
                self.b_max,
                args=(L, s_vectors),
            )
        else:
            res = (
                2
                / (2 * L) ** 2
                * sum(
                    sum(
                        (self.b_max**2 / 2)
                        * (
                            Gaussian_exp(
                                np.linalg.norm(s_vectors[i] - s_vectors[j]) ** 2,
                                2 * self.b_max**2,
                            )
                            + Gaussian_exp(
                                np.linalg.norm(s_vectors[i] + s_vectors[j]) ** 2,
                                2 * self.b_max**2,
                            )
                        )
                        + (1 / 8)
                        * (
                            np.linalg.norm(s_vectors[i] - s_vectors[j]) ** 2
                            * custom_expi(
                                -0.5
                                * np.linalg.norm(s_vectors[i] - s_vectors[j]) ** 2
                                / (2 * self.b_max**2)
                            )
                            + np.linalg.norm(s_vectors[i] + s_vectors[j]) ** 2
                            * custom_expi(
                                -0.5
                                * np.linalg.norm(s_vectors[i] + s_vectors[j]) ** 2
                                / (2 * self.b_max**2)
                            )
                        )
                        for j in range(L)
                    )
                    for i in range(L)
                )
            )
        return res

    def calc_full_dist_symm_even(self, s_vectors, use_integral_D3=None):
        if use_integral_D3 is None:
            use_integral_D3 = self.use_integral_default
        L = len(s_vectors)
        dist = (
            self.D1
            - 2 * self.calc_D2_symm_even(s_vectors)
            + self.calc_D3_symm_even(L, s_vectors, use_integral_D3)
        )
        return dist

    @staticmethod
    def integrand_D2_symm_even_diff(b, L, N, s_vectors, i, d):
        s_i_norm_sq = np.linalg.norm(s_vectors[i]) ** 2
        return (
            -s_vectors[i][d]
            / (2 * L)
            * (2 * b / (1 + 2 * b**2))
            * ((2 * b**2) / (1 + 2 * b**2)) ** (N / 2)
            * Gaussian_exp(s_i_norm_sq, (1 + 2 * b**2))
        )

    @staticmethod
    def integrand_D3_symm_even_diff(b, L, s_vectors, i, d):
        term = 0
        for j in range(L):
            if j != i:
                term += (s_vectors[i][d] - s_vectors[j][d]) * Gaussian_exp(
                    np.linalg.norm(s_vectors[i] - s_vectors[j]) ** 2, 2 * b**2
                ) + (s_vectors[i][d] + s_vectors[j][d]) * Gaussian_exp(
                    np.linalg.norm(s_vectors[i] + s_vectors[j]) ** 2, 2 * b**2
                )
        return -(2 / (2 * L) ** 2) / b * term

    def calc_partial_derivative_D2(self, s_vectors, i, d):
        L = len(s_vectors)
        N = len(s_vectors[0])
        res, _ = quad(
            LCDSymmEvenOptimizer.integrand_D2_symm_even_diff,
            0,
            self.b_max,
            args=(L, N, s_vectors, i, d),
        )
        return res

    def calc_partial_derivative_D3(self, s_vectors, i, d, use_integral=False):
        L = len(s_vectors)
        if use_integral:
            res, _ = quad(
                LCDSymmEvenOptimizer.integrand_D3_symm_even_diff,
                0,
                self.b_max,
                args=(L, s_vectors, i, d),
            )
        else:
            term = 0
            for j in range(L):
                if j != i:
                    term += (s_vectors[i][d] - s_vectors[j][d]) * custom_expi(
                        -0.5
                        * np.linalg.norm(s_vectors[i] - s_vectors[j]) ** 2
                        / (2 * self.b_max**2)
                    ) + (s_vectors[i][d] + s_vectors[j][d]) * custom_expi(
                        -0.5
                        * np.linalg.norm(s_vectors[i] + s_vectors[j]) ** 2
                        / (2 * self.b_max**2)
                    )
            res = (1 / (2 * L) ** 2) * term

        return res

    def calc_full_derivative_symm_even(self, s_vectors, i, d, use_integral_D3=None):
        if use_integral_D3 is None:
            use_integral_D3 = self.use_integral_default
        d2 = self.calc_partial_derivative_D2(s_vectors, i, d)
        d3 = self.calc_partial_derivative_D3(s_vectors, i, d, use_integral_D3)
        return -2 * d2 + d3

    def calc_full_dist_symm_even_wrapper(
        self, reshaped_s_vectors, use_integral_D3=None
    ):
        if use_integral_D3 is None:
            use_integral_D3 = self.use_integral_default
        s_vectors = reshaped_s_vectors.reshape(-1, self.dim)
        return self.calc_full_dist_symm_even(s_vectors, use_integral_D3)

    def calc_full_derivative_symm_even_wrapper(
        self, reshaped_s_vectors, use_integral_D3=None
    ):
        if use_integral_D3 is None:
            use_integral_D3 = self.use_integral_default
        s_vectors = reshaped_s_vectors.reshape(-1, self.dim)
        L = len(s_vectors)
        gradient = np.zeros_like(s_vectors)

        for i in range(L):
            for d in range(self.dim):
                gradient[i, d] = self.calc_full_derivative_symm_even(
                    s_vectors, i, d, use_integral_D3
                )

        return gradient.flatten()

    def get_samples(self, n_samples_half, initial_samples=None):
        if initial_samples is None:
            initial_samples = np.array(
                [np.random.rand(self.dim) for _ in range(n_samples_half)]
            )

        optimization_result = minimize(
            fun=self.calc_full_dist_symm_even_wrapper,
            x0=np.ravel(initial_samples),
            args=(),
            method="L-BFGS-B",
            jac=self.calc_full_derivative_symm_even_wrapper,
            options={"disp": True},
        )
        if not optimization_result.success:
            warnings.warn(
                "Optimization failed. Samples quality may be of questionable quality or totally nonsense."
            )
        samples = np.reshape(optimization_result.x, (-1, self.dim))
        symmetric_samples = np.vstack((samples, -samples))
        return symmetric_samples


if __name__ == "__main__":
    b_max_test = 5  # Set appropriately
    n_samples_half_test = 30
    dim_test = 2

    lcd_opt = LCDSymmEvenOptimizer(b_max_test, dim_test)

    # Fix the random seed for deterministic test c ase
    np.random.seed(42)
    test_samples = np.array(
        [np.random.rand(dim_test) for _ in range(n_samples_half_test)]
    )

    # Test individual components for distance
    print("D1 is:", lcd_opt.D1)

    result = lcd_opt.calc_D2_symm_even(test_samples)
    print("D2 is:", result)

    result = lcd_opt.calc_D3_symm_even(n_samples_half_test, test_samples, True)
    print("D3 is:", result)

    result = lcd_opt.calc_D3_symm_even(n_samples_half_test, test_samples, False)
    print("D3 is:", result)

    # Test distance
    result = lcd_opt.calc_full_dist_symm_even(test_samples, True)
    print("The modified Cramer-von Mises Distance (using integral for D3) is:", result)

    result = lcd_opt.calc_full_dist_symm_even(test_samples, False)
    print("The modified Cramer-von Mises Distance (using expi) is:", result)

    # Test individual components for derivative
    i_test = 5
    d_test = 1
    result = lcd_opt.calc_partial_derivative_D2(test_samples, i_test, d_test)
    print("dD2 is:", result)

    result = lcd_opt.calc_partial_derivative_D3(test_samples, i_test, d_test, False)
    print("dD3 is (using integral):", result)

    result = lcd_opt.calc_partial_derivative_D3(test_samples, i_test, d_test, False)
    print("dD3 is (using expi):", result)

    # Test derivative

    result = lcd_opt.calc_full_derivative_symm_even(test_samples, i_test, d_test, False)
    print("The full partial derivative is (using integral):", result)

    result = lcd_opt.calc_full_derivative_symm_even(test_samples, i_test, d_test, True)
    print("The full partial derivative is (using expi):", result)

    initial_guess = test_samples.flatten()

    # Test generation of samples
    samples = lcd_opt.get_samples(n_samples_half_test)
    print("Generated samples:", samples)

    test_mean = np.mean(samples, axis=0)
    print("Mean of symmetrized samples:", test_mean)

    test_cov = np.cov(samples, rowvar=False)
    print("Cov of symmetrized samples:", test_cov)

    cov_chol = np.linalg.cholesky(test_cov)

    standardized_symm_samples = np.linalg.solve(cov_chol, samples.T).T

    print("Standardized samples:", standardized_symm_samples)

    test_stand_cov = np.cov(standardized_symm_samples, rowvar=False)

    print("Cov of standardized samples:", test_stand_cov)
