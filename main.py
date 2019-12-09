import numpy as np 
import imageio
import time
import os
from skimage import img_as_float, exposure, measure
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler

if __name__ == '__main__':
    
    data_dir = '../visao/data-2/'
    classes = ['apple', 'chicken']
    #CRIA O VETOR DE CARACTERISTICAS
    caracteristicas = []
    labels = []
    rotulo = 0

    for directory in classes:

        data = os.listdir(data_dir+directory)
        data.sort()
        

        for arquivo in data:

            img = imageio.imread(data_dir + directory + '/' + arquivo)
            img_np = np.array(img)
            print(" ")
            print(arquivo)
            
            #CONVERTENDO PARA FLOAT
            img_float = img_as_float(img_np)

            #GERA IMAGEM DE ROTULOS
            img_r = measure.label(img, background=0)
            #GERA OS PROPS
            props = measure.regionprops(img_r, img)

            #PRA CADA OBJETO DETECTADO NA IMAGEM ROTULADA, PRINTA AS CARACTERISTICAS DESEJADAS
            for i in range (0, img_r.max()):
                print('Objeto: ', i)
                print('Area ..............: ', props[i].area)
                print('Excentricidade ....: ', props[i].eccentricity)
                print('Extensao ..........: ', props[i].extent)
                print('Solidez ...........: ', props[i].solidity)
                print(' ')
                print('---------------------------------')
                caracteristicas.append(props[i].area)
                caracteristicas.append(props[i].eccentricity)
                caracteristicas.append(props[i].extent)
                caracteristicas.append(props[i].solidity)
                time.sleep(0.4)

            labels.append(rotulo)
        rotulo += 1

    
    #TRANSFORMO LISTA DE CARACTERISTICAS E ROTULOS EM UM NP ARRAY
    caracteristicas = np.asarray(caracteristicas)
    print(caracteristicas)

    labels = np.asarray(labels)
    #TRANSFORMO O VETOR EM UMA MATRIZ 16X4 (16 OBJETOS X 4 CARACTERISTICAS)
    data = caracteristicas.reshape(len(labels), 4)

    print("\nMATRIZ DE OBJETOS \n")
    print(data)
    print("\nVETOR DE ROTULOS")
    print(labels)

    #NORMALIZANDO AS CARACTERISTICAS
    transformer = MaxAbsScaler().fit(data)   
    data = transformer.transform(data)

    print("\nCARACTERISTICAS NORMALIZADAS")
    print(data)

    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.25, random_state=42)

    print('\nCONJUNTO DE TREINAMENTO')
    print(xtrain)
    print('\nCONJUNTO DE TESTE')
    print(xtest)

    print('\nROTULOS DE TREINAMENTO')
    print(ytrain)
    print('\nROTULOS DE TESTE')
    print(ytest)

    classificadores = ['knn', 'svm', 'bayes']

    for clf in classificadores:

        if(clf == 'knn'):
            print("\n============ KNN ============\n")
            classificador = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        if(clf == 'bayes'):
            print("\n============ Bayes ============\n")
            classificador = GaussianNB()
        if(clf == 'svm'):
            print("\n============ SVM ============\n")
            classificador = SVC()

        classificador.fit(xtrain, ytrain)
        pred = classificador.predict(xtest)
        print('Predição:', pred)
        print('    Real:', ytest)

        print('Matriz de confusão:')
        print(metrics.confusion_matrix(ytest, pred))

        print('\nRelatório de classificação:')
        print(metrics.classification_report(ytest, pred))

        print('Acuracia:', 100*metrics.accuracy_score(ytest, pred), '% \n\n')
        time.sleep(0.4)
