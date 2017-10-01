def gdmulti(X, y, theta, alpha, num_iters):
    XMatrix = np.mat(X)
    yMatrix = np.mat(y)
    thetaMatrix = np.mat(theta)
    m = XMatrix.shape[0]
    J_hist = np.zeros((num_iters, 1))
    for i in range(num_iters):
        thetaMatrix = thetaMatrix - XMatrix.T * (XMatrix * thetaMatrix - yMatrix) * alpha / m

        J_hist[i, :] = (computarCustoMulti(X, y, thetaMatrix, m))

    return thetaMatrix, J_hist