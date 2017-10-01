def normalizarCaracteristica(X, comprimento):
    """ Função  normalizarCaracteristica 
    -> Subtrai o valor médio de todas as características do conjunto de dados.
    -> Após subtrair a média, divide cada característica pelos seus respectivos
        desvios padrões.
A função normalizarCaracteristica deve receber a matriz de dados X de dados 
como parâmetro (na forma de um numpy array). Além disso, essa função
deve funcionar com conjuntos de dados de variados tamanhos (qualquer quan-
tidade de características / exemplos).  

Também armazena os valores de valor médio e desvio
padrão utilizados para a normalização. 
    
    """
    norx = X.copy()
    
    # guarda o valor temporario da média de X
    valor_media = np.zeros((1, norx.shape[1]))
    
    # guarda o valor temporario do desvio padrão de X
    desvio_padrao = np.zeros((1, norx.shape[1]))


    valor_media = np.mean(norx, axis=0).reshape((1, 2))
    # numpy: no desvio padrão deve ser adicionado ddof=1,ou usar o módulo statistics
    # encontrada essa solução em consulta a
    # https://stackoverflow.com/questions/15389768/standard-deviation-of-a-list
    desvio_padrao = norx.std(axis=0, ddof=1).reshape((1, 2))

    norx = (norx - np.tile(valor_media, (comprimento, 1))) / np.tile(desvio_padrao, (comprimento, 1))
         
    return norx, valor_media, desvio_padrao