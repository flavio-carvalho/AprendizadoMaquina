def learningCurve(X, y, Xval, yval, reg):
    m = y.size
    erro_treino = np.zeros((m, 1))
    erro_val = np.zeros((m, 1))
    for i in np.arange(m):
        res = trainLinearReg(X[:i+1], y[:i+1], reg)
        erro_treino[i] = linearRegCostFunction(res.x, X[:i+1], y[:i+1], reg)
        erro_val[i] = linearRegCostFunction(res.x, Xval, yval, reg)
    
    return(erro_treino, erro_val)
