def plotDecisionBoundary():

    fig, axes = plt.subplots(1,2, sharey = True, figsize=(17,5))

    """
    Serão gerados dois gráficos com a fronteira de decisão que foi aprendida
    para separar os exemplos positivos dos negativos.

    Fronteiras de decisão:
    Sem regularização: Lambda = 0 
    Regularizado com Lambda = 1 
    """
    for i, C in enumerate([0,1]):
        # Valores ótimos da função costFunctionReg
        res2 = minimize(costFunctionReg, initial_theta, args=(C, XX, y), method=None, jac=gradientReg, options={'maxiter':3000})
    
        # Acurácia (precisão)
        accuracy = 100*sum(predict(res2.x, XX) == y.ravel())/y.size    

        # Scatter plot of X,y
        plotar(dados2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0',axes.flatten()[i])
    
        # Plot decisionboundary
        x1_min, x1_max = X[:,0].min(), X[:,0].max(),
        x2_min, x2_max = X[:,1].min(), X[:,1].max(),
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
        h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
        h = h.reshape(xx1.shape)
        axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
        axes.flatten()[i].set_title('Lambda = {}'.format(C))
