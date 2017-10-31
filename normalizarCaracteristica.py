def normalizarCaracteristica(X, mu=None, sigma=None):
    """
    Função  normalizarCaracteristica 
    -> Subtrai o valor médio de todas as características do conjunto de dados.
    -> Após subtrair a média, divide cada característica pelos seus respectivos
        desvios padrões.
    A função normalizarCaracteristica deve receber a matriz de dados X de dados 
    como parâmetro (na forma de um numpy array). Além disso, essa função
    deve funcionar com conjuntos de dados de variados tamanhos (qualquer quan-
    tidade de características / exemplos).  

    Também armazena os valores de valor médio e desvio
    padrão utilizados para a normalização. 


    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples to be normalized, where n_samples is the number of samples and n_features is the number of features.
    mu : ndarray, shape (n_features,)
        Mean value for normalization. If not provided, it will be calculated from X.
    sigma : ndarray, shape (n_features,)
        Standard deviation for normalization. If not provided, it will be calculated from X.

    Returns
    -------
    X_norm : ndarray, shape (n_samples, n_features)
        The normalized features.
    mu : ndarray, shape (n_features,)
        Mean value of X.
    sigma : ndarray, shape (n_features,)
        Standard deviation of X.
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm/sigma

    return X_norm, mu, sigma
