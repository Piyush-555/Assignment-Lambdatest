import math
import numpy as np

class Simple_ES:
    """
    A simple (mu + lambda) type evolution strategy.
    Uses lambda set of candidates to explore and select top-mu of them.
    """
    
    def __init__(self, x0, mu=None, lambdaa=None, sigma=1):
        assert x0.__class__ is np.ndarray
        self.num_params = x0.shape[0]
        
        if lambdaa is None:
            self.lambdaa = 4 + int(np.log(self.num_params))
            self.mu = math.ceil(0.25 * self.lambdaa)
        else:
            self.lambdaa = lambdaa
            if mu is None or mu > self.lambdaa:
                self.mu = math.ceil(0.25 * self.lambdaa)
            else:
                self.mu = mu
        
        self.mean = x0.copy()
        self.sigma = sigma

    def ask(self):
        solutions = []
        for i in range(self.lambdaa):
            a_soln = np.random.normal(loc=self.mean, scale=self.sigma, size=self.mean.shape)
            solutions.append(a_soln)
        return solutions
    
    def tell(self, solutions, function_values):
        sorted_solutions = [soln for _, soln in sorted(zip(function_values, solutions), key=lambda x: x[0])]
        elites = np.array(sorted_solutions[:self.mu])
        self.mean = elites.mean(axis=0)
