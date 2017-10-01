def funcaoCustoRegressaoLogistica(theta,X, y,retornagradiente=False):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    a = np.multiply(-y, np.log(sigmoide(X * theta.T)))
    b = np.multiply((1 - y), np.log(1 - sigmoide(X * theta.T)))
    resultado = np.sum(a - b) / (len(X))
    
    # Se não for solicitado que seja retornado o valor do gradiente
    # com retornagradiente = 1, a função vai retornar somente
    # o valor do custo.
    if retornagradiente==True:
        return resultado, gradiente(theta,X, y)
    else:
        return resultado