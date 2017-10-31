def estimativaGaussian(X):
    """
    Esta funcao toma como entrada a matriz de dados X e deve produzir
    um vetor de dimensao n mu que contem a media de todas as caracteristicas n
    e outro vetor de dimensao n sigma2 que contem as variancias de todas as caracteristicas.
    """
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma
