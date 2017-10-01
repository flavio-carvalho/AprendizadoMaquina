def computarCusto(X, y, theta):
    """
    computarCusto(X, y, theta) calcula o custo usando theta como 
    parametro da regressão linear para ajustar os pontos em X e y
    Retorna o valor da função de custo.
    theta -- Gradiente Descendente
    """
    return np.power(h(X, theta) - y, 2).mean()/2