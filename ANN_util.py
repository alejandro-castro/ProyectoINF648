#Elaborado por Alejandro Castro
#Todo el código presente ha sido elaborado desde cero por el autor con la excepción de la parte de ploteo del método fitCV, 
#que fue adaptada de código tomado de la especialización de Deep Learning en Coursera.

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def generateKFoldValidationSetWithPreprocessing(x_train, y_train, cv=8, shuffle=True, randomState=0):
    """
    Función que genera un conjunto de tuplas (xfold, yfold, xdev, ydev) a partir de un conjunto de entrenamiento
    aplicando validación cruzada K-Fold, e imputación y escalamiento para cada una de las tuplas.
    """
    
    #Generación de los conjuntos de validación cruzada
    cross_validator = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=randomState)

    cross_validation_sets = []
    for train_index, dev_index in cross_validator.split(x_train, y_train):
        x_fold, y_fold,  = x_train.iloc[train_index,:], y_train.iloc[train_index]
        x_dev, y_dev= x_train.iloc[dev_index,:], y_train.iloc[dev_index]

        #Imputación
        ann_imputer = SimpleImputer(strategy="mean").fit(x_fold)
        x_imputed = ann_imputer.transform(x_fold)
        x_dev_imputed = ann_imputer.transform(x_dev)

        #Escalamiento
        ann_scaler = StandardScaler().fit(x_imputed)
        x_fold_scaled = ann_scaler.transform(x_imputed)
        x_dev_scaled = ann_scaler.transform(x_dev_imputed)

        cross_validation_sets.append((x_fold_scaled, y_fold, x_dev_scaled, y_dev))
        
    return cross_validation_sets
    
    
class ANNArchitecture():
    """
    Clase de abstracción de arquitectura de redes neuronales totalmente conectadas(densas), se desarrolló para facilitar el
    crear arquitectura en ejecución mediante el uso de lista simples.
    
    Las instancias de la clase se crean con dos listas: unitsList que contiene la cantidad de neuronas de cada capa oculta o de
    salida y activationsFunctionsList que contiene la función de activación para cada una de las capas correspondientes.
    """
    
    def __init__(self, unitsList, activationsFunctionsList):
        self.numberOfUnits = unitsList
        self.activationsFunctions = activationsFunctionsList
        self.numberOfLayers = len(unitsList)
        
    def getNumberOfUnits(self, layer):
        return self.numberOfUnits[layer]
    
    def getActivationFunction(self, layer):
        return self.activationsFunctions[layer]
    
    def getNumberOfLayers(self):
        return self.numberOfLayers
    
    
class FeedForwardNeuralNetwork():
    """
    Clase que permite el entrenamiento de una red neuronal usando validación cruzada y múltiples inicializaciones Xavier de los
    pesos, lo que permite lograr una mejor exactitud de validación
    """
    
    def __init__(self, architecture):
        """
        Los atributos de esta clase son los siguientes:
        architecture: Instancia de ANNArchitecture que permite crear un modelo de Tensorflow en tiempo real de manera fácil.
        model: Modelo sequential de tensorflow, es usado para guardar el modelo entrenado final.
        imputer: Instancia de SimpleImputer() de Scikit-Learn que fue ajustada con toda la data, permite que esta clase pueda
        recibir data sin imputación previa.
        scaler: Instancia de StandardScaler de Scikit-Learn que fue ajustada con toda la data, permite que esta clase pueda
        recibir data sin escalamiento previo.
        """
        self.architecture = architecture
        self.model = None
        self.imputer = None
        self.scaler = None
        
        
    def _createModel(self, numberOfFeatures):
        """
        Crea un modelo de Tensorflow para la arquitectura de la instancia, este modelo utiliza una inicialización de Xavier
        uniforme.

        Parámetros:
        numberOfFeatures: Cantidad de características, es utilizada para crear la capa de entrada de la red neuronal.

        Devuelve:
        model: un modelo Sequential de Tensorflow inicializado.
        """
        #Creando el tensor de entrada
        inputs = tf.keras.Input(shape=(numberOfFeatures,))

        #Creando el modelo
        model = tf.keras.Sequential() #(inputs=inputs, outputs=outputs)
        model.add(inputs)

        #Iteración sobre las capas de la arquitectura para generar el modelo correspondiente
        for layer in range(self.architecture.getNumberOfLayers()):
            numberOfUnits = self.architecture.getNumberOfUnits(layer)
            activation = self.architecture.getActivationFunction(layer)
            model.add(tf.keras.layers.Dense(numberOfUnits, activation=activation, use_bias=True,
                                            kernel_initializer=tf.keras.initializers.GlorotUniform, 
                                            bias_initializer=tf.zeros_initializer))
            # Agregar si se desea agregar regularización por capa, kernel_regularizer=tf.keras.regularizers.l2(l=0.005)
                                            
    
        return model
 

    def _fit(self, X_train, Y_train, learningRate=0.0001, numberOfEpochs=1500, minibatchSize=32, verbose=True,
            numberOfInitializations=20, cv=8, randomState=0):
        """
        Método que permite entrenar el modelo dado por la arquitectura de la clase con una determinada data y configuración.
        El entrenamiento del modelo utiliza Early Stopping para evitar el sobre ajuste.
        Asimismo el modelo se entrena con distintas inicializaciones aleatorias y se escoje la mejor utilizando validación
        cruzada.
        Para el entrenamiento siempre se utiliza un optimizador de Adam y la entropía cruzada categórica dispersa como función
        de pérdida.

        Parámetros:
        X_train: Conjunto de características de entrenamiento
        Y_train: Conjunto de etiquetas para las características de entrenamiento dadas, el número de clases presentes debe
        coincidir con el neuronas de salida definidas en la arquitectura.

        learningRate: Tasa de aprendizaje que utilizará el optimizador
        numberOfEpochs: Máximo número de épocas de entrenamiento, el número de épocas real puede ser menor debido al Early
        Stopping.
        minibatchSize: Tamaño del minibatch utilizado durante el entrenamiento
        verbose: Booleano que permite decidir si se desea imprimir información o no. 
        True: mostrar información, False: no mostrar información.
        
        numberOfInitializations: Número de inicializaciones que se realizarán para la misma arquitectura.
        cv: Número k folds que se utilizarán en la validación cruzada, cv debe ser mayor a dos, si cv es None quiere decir que
        se desea entrenar con toda la data.
        randomState: Semilla aleatoria para ser usada durante el entrenamiento
        
        Devuelve:
        El history del modelo entrenado si es cv es None, en cambio si cv >=2, no se devuelve nada.
        """
        
        assert (cv is None) or(cv >= 2)
        
        numberOfFeatures = X_train.shape[1]

        #Fijamos la semilla aleatoria de Tensorflow para volver reproducible el código
        tf.random.set_seed(randomState) 

        #Definición del optimizador
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        
        #Definición de la función de pérdida o costo
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        shouldValidate = cv is not None 
        #Entrenamiento de varios modelos con misma arquitectura pero con diferente inicialización aleatoria
        if shouldValidate: 
            #Obteniendo los conjuntos de validación cruzada
            cross_validation_sets = generateKFoldValidationSetWithPreprocessing(X_train, Y_train, cv=cv, shuffle=True,
                                                                                randomState=0)
            
            #Callback que monitorea el costo de validación o entrenamiento para aplicar detener el entrenamiento 
            #si no hay mejoras significativas
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=10**-3, patience=5, verbose=0,
                                                        mode='min')

            modelsAndMetrics = [] #Lista para almacenar los distintos modelos entrenados así como sus métricas

            for i in range(numberOfInitializations):
                #Creación del modelo
                model = self._createModel(numberOfFeatures)

                averageAccuracy = 0
                averageValidationAccuracy = 0
                iteration = 1
                
                for x_fold_scaled, y_fold, x_dev_scaled, y_dev in cross_validation_sets:
                    #Copia del modelo, necesaria para aplicar validación cruzada a la inicialización
                    new_model = self._createModel(numberOfFeatures)
                    for j in range(len(model.layers)):
                        weights = model.layers[j].get_weights()
                        new_model.layers[j].set_weights(weights)

                    #Compilando el modelo
                    new_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

                    history = new_model.fit(x=x_fold_scaled, y=y_fold, batch_size=minibatchSize, epochs=numberOfEpochs,
                                            verbose=0, shuffle=True, validation_data=(x_dev_scaled, y_dev), 
                                            callbacks=[callback], use_multiprocessing=True)     

                    averageAccuracy += history.history["accuracy"][-1]
                    averageValidationAccuracy += history.history["val_accuracy"][-1]
                    iteration += 1

                averageAccuracy /= cv
                averageValidationAccuracy /= cv

                modelsAndMetrics.append({"model":model, "accuracy":averageAccuracy, "val_accuracy":averageValidationAccuracy})
                if verbose:
                    print(f"Inicialización {i+1}")
                    print(f"Con esta inicialización se logró una exactitud de validación promedio: {averageValidationAccuracy}"\
                          f" y una exactitud de entrenamiento promedio: {averageAccuracy}.\n")
            
            #Guardamos el modelo con la mejor inicialización, modelo sin entrenar
            self.model, bestValidationAccuracy = self.getHighestAccuracyModel(modelsAndMetrics, "val_accuracy")   
            print(f"La mejor inicialización para esta arquitectura logró una exactitud de validación promedio: "\
                  f"{averageValidationAccuracy} y una exactitud de entrenamiento promedio: {averageAccuracy}.\n")
        else:
            #Callback que monitorea el costo de validación o entrenamiento para aplicar detener el entrenamiento 
            #si no hay mejoras significativas
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=10**-3, patience=5, verbose=0, mode='min')
            
            #Compilando el modelo guardado, si no se crea
            if self.model is None:
                self.model = self._createModel(numberOfFeatures)
            self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
            
            #Imputacion
            ann_imputer = SimpleImputer(strategy="mean").fit(X_train)
            x_imputed = ann_imputer.transform(X_train)
            self.imputer = ann_imputer

            #Escalamiento
            ann_scaler = StandardScaler().fit(x_imputed)
            x_train_scaled = ann_scaler.transform(x_imputed)
            self.scaler = ann_scaler

            history = self.model.fit(x=x_train_scaled, y=Y_train, batch_size=minibatchSize, epochs=numberOfEpochs, verbose=0,
                                     shuffle=True, validation_data=None, callbacks=[callback], use_multiprocessing= True)
            return history.history

        
    def fitCV(self, X_train, Y_train, learningRate=0.0001, numberOfEpochs=1500, minibatchSize=32, verbose=True,
            numberOfInitializations=20, cv=8, randomState=0):
        """
        Método que busca el mejor modelo mediante la validación cruzada del método _fit y que luego reentrena el modelo 
        encontrado con toda la data de entrenamiento.

        Parámetros:
        X_train: Conjunto de características de entrenamiento
        Y_train: Conjunto de etiquetas para las características de entrenamiento dadas, el número de clases presentes debe
        coincidir con el neuronas de salida definidas en la arquitectura.

        learningRate: Tasa de aprendizaje que utilizará el optimizador
        numberOfEpochs: Máximo número de épocas de entrenamiento, el número de épocas real puede ser menor debido al Early
        Stopping.
        minibatchSize: Tamaño del minibatch utilizado durante el entrenamiento
        verbose: Booleano que permite decidir si se desea imprimir información o no. 
        True: mostrar información, False: no mostrar información.
        
        numberOfInitializations: Número de inicializaciones que se realizarán para la misma arquitectura.
        cv: Número k folds que se utilizarán en la validación cruzada, cv debe ser mayor a dos, si cv es None quiere decir que
        se desea entrenar con toda la data.
        randomState: Semilla aleatoria para ser usada durante el entrenamiento
        
        Devuelve:
        No devuelve nada, solo guarda el modelo entrenado.
        """
        
        #Arreglos utilizados para plotear, de ser requerido
        costsToPlot = []                                       
        valCostsToPlot = []
        
        startTime = time.time()
        self._fit(X_train, Y_train, learningRate, numberOfEpochs, minibatchSize, verbose, numberOfInitializations, cv,
                  randomState)
        
        print("\nEntrenamiento del modelo que mejor exactitud de validación promedio tuvo usando toda la data de "\
        "entrenamiento.")
        history = self._fit(X_train, Y_train, learningRate, numberOfEpochs, minibatchSize, verbose,
                                                numberOfInitializations, cv=None, randomState=randomState)
        
        endTime = time.time()

        #Guardando los costos y las exactitudes
        costs = history["loss"]
        accuracies = history["accuracy"]

        trainDuration = len(costs) #Ya no es numberOfEpochs, porque hemos aplicado early stopping
        print(f"Tiempo Transcurrido:{time.strftime('%H:%M:%S', time.gmtime(endTime - startTime))}. Número de épocas:"\
              f"{trainDuration}.\n")

        #Si se desea mas información se imprime los costos cada 100 o 10 epochs(depende del total de epochs), y luego se
        #grafican los costos.
        if verbose:
            for epoch in range(trainDuration):
                printStep =  100 if trainDuration>=100 else 10 #Cálculo de cada cuanto se imprime
                if (epoch % printStep == 0 or epoch == trainDuration - 1):
                    print (f"""Después de la época {epoch}: cost:{costs[epoch]}, acc:{accuracies[epoch]}""")

                if epoch % 5 == 0:
                    costsToPlot.append(costs[epoch])

            #Graficando el costo
            plt.plot(costsToPlot, label="Costo de entrenamiento")
            plt.legend()

            plt.ylabel('Costo')
            plt.xlabel('Número de épocas (cada cinco)')
            plt.title(f"Tasa de aprendizaje ={learningRate}, número de épocas:{trainDuration}")
            plt.show()
    
        print (f"Costo de entrenamiento:{costs[-1]}. Exactitud de entrenamiento:{accuracies[-1]}.\n\n")


    def score(self, x, y):
        """
        Método que devuelve la precisión del modelo para el correspondiente par de datos x,y
        Parámetros:
        x,y : Conjunto de características y etiquetas respectivamente, pueden ser Dataframes, Series de Pandas respectivamente o
        arreglos de Numpy, lo único necesario es que x tenga la forma (número de muestras, número de características)
        """
        x_imputed = self.imputer.transform(x)
        x_scaled = self.scaler.transform(x_imputed)
        return self.model.evaluate(x=x_scaled, y=y, batch_size=64, verbose=0, return_dict=True)["accuracy"]

    
    def getHighestAccuracyModel(self, modelsAndMetrics, metric="val_accuracy"):
        """
        Método que a partir de una lista de tuplas (modelo, métrica entrenamiento, métrica de validación), devuelve el modelo
        que al que le corresponde el menor valor de la métrica dada por metric

        Parámetros:
        modelsAndMetrics: Lista de diccionarios {modelo:, accuracy:, val_accuracy:} que corresponden a entrenamientos de modelos
        con distintos pesos iniciales pero con una misma arquitectura.
        metric: Métrica a maximizar en la búsqueda dentro de la lista de modelos. Por el momento solo se ha probado 
        "val_accuracy" y "accuracy"

        Devuelve:
        El mejor modelo encontrado y su respectiva métrica
        """

        indexOfHighestMetric = 0
        highestMetric = modelsAndMetrics[indexOfHighestMetric][metric]

        index = 0
        for modelAndMetric in modelsAndMetrics:
            currentMetric = modelAndMetric[metric]
            if currentMetric > highestMetric:
                highestMetric = currentMetric
                indexOfHighestMetric = index

            index += 1

        return modelsAndMetrics[indexOfHighestMetric]["model"], modelsAndMetrics[indexOfHighestMetric][metric]
