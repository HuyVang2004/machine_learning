import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score

class MySVM:
    def __init__(self, C=1e6, kernel='rbf', degree=3, coef0=1, gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.lambdas = None  
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None

    def linear_kernel(self, X, Z):
        return np.dot(X, Z.T)

    def polynomial_kernel(self, X, Z):
        return (self.gamma * np.dot(X, Z.T) + self.coef0) ** self.degree

    def sigmoid_kernel(self, X, Z):
        return np.tanh(self.gamma * np.dot(X, Z.T) + self.coef0)
    
    def rbf_kernel(self, X, Z):
        X_norm = np.sum(X ** 2, axis=-1)
        Z_norm = np.sum(Z ** 2, axis=-1)
        K = -2 * np.dot(X, Z.T) + X_norm[:, np.newaxis] + Z_norm[:, np.newaxis]
        return np.exp(-self.gamma * K)
    
    def kernel_function(self, X, Z):
        if self.kernel == "linear":
            return self.linear_kernel(X, Z)
        elif self.kernel == "poly":
            return self.polynomial_kernel(X, Z)
        elif self.kernel == "sigmoid":
            return self.sigmoid_kernel(X, Z)
        elif self.kernel == "rbf":
            return self.rbf_kernel(X, Z)
        else:
            raise ValueError("Invalid kernel. Choose between {'poly', 'sigmoid', 'linear', 'rbf'}.")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if self.gamma == 'scale':
            self.gamma = 1 / (n_features * np.var(X))
        elif self.gamma == 'auto':
            self.gamma = 1 / n_features
        
        K = self.kernel_function(X, X)

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))

        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.vstack((np.zeros((n_samples,1)), np.ones((n_samples, 1)) * self.C)))

        A = matrix(y, (1, n_samples))
        b = matrix(0.0)

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        lambdas = np.ravel(solution['x'])

        S = np.where(lambdas > 1e-6)[0] 
        S2 = np.where(lambdas < 0.999 * self.C)[0]

        support_vector_indices = [val for val in S if val in S2]
        self.lambdas = lambdas[support_vector_indices]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]

        self.b = np.mean(
            [y_i - np.sum(self.lambdas * self.support_vector_labels * self.kernel_function(self.support_vectors, x_i)) 
             for x_i, y_i in zip(self.support_vectors, self.support_vector_labels)]
        )

    def predict(self, X):
        K = self.kernel_function(self.support_vectors, X)  
        y_predict = np.dot(self.lambdas * self.support_vector_labels, K) + self.b  
        return np.sign(y_predict)


if __name__ == '__main__':
    means = [[1, 5, 2], [1, 4, 5]] 
    cov1 = [[1, 0.3, 0.2], [0.3, 1, 0.4], [0.2, 0.4, 1]] 
    cov2 = [[1.5, -0.4, 0.1], [-0.4, 1.2, 0.5], [0.1, 0.5, 1.8]]

    N = 150

    X1 = np.random.multivariate_normal(mean=means[0], cov=cov1, size=N)
    X2 = np.random.multivariate_normal(mean=means[1], cov=cov2, size=N)

    X = np.concatenate((X1, X2), axis=0)

    y = np.concatenate((np.ones(N), -np.ones(N)))

    svm = MySVM(C = 2, degree=15, kernel='poly')

    svm.fit(X, y)

    pred = svm.predict(X)
    print('accuracy', accuracy_score(pred, y))