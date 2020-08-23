from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class EncoderXGBoost(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.enc = OrdinalEncoder()
        # Encoder que transforma os nomes das classes em valores numéricos que serão melhores interpretados pelo classificador.
        self.enc.fit([['EXCELENTE'], ['MUITO_BOM'], ['HUMANAS'], ['EXATAS'], ['DIFICULDADE']])

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'Y' de entrada.
        data = X.copy()
        # Codificamos para um novo dataframe
        data['PERFIL'] = self.enc.transform(data)
        # Retornamos um novo dataframe com os valores codificados.
        return data

class DecoderXGBoost(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.enc = OrdinalEncoder()
        # Encoder que transforma os nomes das classes em valores numéricos que serão melhores interpretados pelo classificador.
        self.enc.fit([['EXCELENTE'], ['MUITO_BOM'], ['HUMANAS'], ['EXATAS'], ['DIFICULDADE']])

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro extraímos a classe que teve o melhor fit para cada resultado.
        best_preds = np.asarray([[np.argmax(line) for line in X]]).transpose()
        # Decodificamos para os tipos de dados originais da coluna de PERFIL (object).
        best_preds = self.enc.inverse_transform(best_preds)
        # Retornamos um np.array com as novas predições.
        return best_preds