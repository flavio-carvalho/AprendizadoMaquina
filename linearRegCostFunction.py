def linearRegCostFunction(theta, X, y, reg):
    m = y.size
    h = X.dot(theta)
    J = (1/(2*m))*np.sum(np.square(h-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    return(J)

# gradiente da regressao linear regularizada
def lrgradientReg(theta, X, y, reg):
    m = y.size
    h = X.dot(theta.reshape(-1,1))
    grad = (1/m)*(X.T.dot(h-y))+ (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
    return(grad.flatten())
