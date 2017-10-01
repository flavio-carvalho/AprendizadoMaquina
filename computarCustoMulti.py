def computarCustoMulti(X, y, thetaMatrix, m):
    XMatrix = np.mat(X)
    yMatrix = np.mat(y)

    J = sum(np.array(XMatrix * thetaMatrix - yMatrix) ** 2) / (2 * m)
    return J