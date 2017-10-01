def gduni(x, y, theta, alpha, numerodeiteracoes):
    """
    gduni Executa o algoritmo Gradiente Descendente
    (gd, custo) = gduni(x, y, theta, alpha, numerodeiteracoes)  

    """
    thetas = [theta]
    custos = []
    for i in range(numerodeiteracoes):
        delta = alpha * np.multiply(h(x, theta)-y, x).mean(0)
        theta = theta - delta
        thetas.append(theta)
        custo = computarCusto(x, y, theta)
        custos.append(custo)
    return thetas, custo