import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

class FashionMNISTModel:
    def __init__(self):
        self.dataset = keras.datasets.fashion_mnist
        self.model = None
        self.history = None
        self.load_data()
        self.preprocess_data()
        self.build_model()
    
    def load_data(self):
        (self.imagens_treino, self.identificadores_treino), (self.imagens_teste, self.identificadores_teste) = self.dataset.load_data()
    
    def preprocess_data(self):
        self.imagens_treino = self.imagens_treino / 255.0
        self.imagens_teste = self.imagens_teste / 255.0
    
    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
    
    def train(self, epochs=5, validation_split=0.2):
        self.history = self.model.fit(
            self.imagens_treino, 
            self.identificadores_treino, 
            epochs=epochs, 
            validation_split=validation_split
        )
    
    def save_model(self, filename="modelo.h5"):
        self.model.save(filename)
    
    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Acurácia do Modelo')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend(['Treino', 'Validação'])
        plt.show()
    
    def test_model(self):
        perda_teste, acuracia_teste = self.model.evaluate(self.imagens_teste, self.identificadores_teste)
        print("Perda do teste:", perda_teste)
        print("Acurácia do teste:", acuracia_teste)
    
    def predict(self, index):
        predicoes = self.model.predict(self.imagens_teste)
        print("Resultado do teste:", np.argmax(predicoes[index]))
        print("Número da imagem teste:", self.identificadores_teste[index])

if __name__ == "__main__":
    fashion_mnist = FashionMNISTModel()
    fashion_mnist.train(epochs=5)
    fashion_mnist.save_model()
    fashion_mnist.plot_accuracy()
    fashion_mnist.test_model()
    fashion_mnist.predict(2)
