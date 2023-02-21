import numpy as np



if __name__ == '__main__':
    print(np.random.default_rng().dirichlet((10, 5, 3), 10))
    