def cofiCostFunc(params, Y, R, num_features):  
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    
    # Calcula o custo J
    error = np.multiply((X * Theta.T) - Y, R)  
    squared_error = np.power(error, 2)
    J = (1. / 2) * np.sum(squared_error)
    
    # Calcula o gradiente
    X_grad = error * Theta
    Theta_grad = error.T * X
    
    # Arrumacao das matrizes
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
    
    return J, grad
