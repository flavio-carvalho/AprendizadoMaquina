def mapFeature(x1, x2, degree=6):
    map_feature_size = x1.size
    map_feature = np.ones(shape=(map_feature_size, 1))
    
    for i in range(0, degree + 1):
        for j in range(i + 1):
            """ Nesse laço, são inseridos os valores ao fim do vetor """
            column = (x1 ** (i - j)) * (x2 ** j)
            map_feature = np.append(map_feature, column, axis=1)
        
    return mapfeature
