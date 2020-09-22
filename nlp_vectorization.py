from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
import gensim


class Vectorization():
    """
    Класс для векторизации текста. Содержит в себе пока два метода: tf-idf и word2vec.
    Класс содержит в себе методы fit и transform для использования объектов класса в пайплайнах sklearn.
    """
    def __init__(self, kind_vector='tf_idf', word_2_vec_model=None, **kwargs):
        """
        Выбор метода векторизацаии
            kind_vector: {'tf_idf', 'word2vec'}, default='tf_idf'
                Вид векторизации - tf_idf или word2vec
            word_2_vec_model: word2vec model, default=None
                Готовая обученная модель word2vec при kind_vector='word2vec'
            **kwargs
                Аргументы для моделей векторизации
        """
        self.kind_vector = kind_vector
        if kind_vector == 'tf_idf':
            self.vectorizer = TfidfVectorizer(**kwargs)
        elif kind_vector == 'word2vec':
            self.vectorizer = word_2_vec_model
    
    def fit(self, X, y=None):
        """
        Метод для обучения модели векторизации
            X: pandas.DataFrame/numpy.ndarray
                Датасет (корпус текста)
        """
        self.X_init = X
        
        if self.kind_vector == 'tf_idf':
            self.vectorizer.fit(X)
        elif self.kind_vector == 'word2vec':
            self
    
    def transform(self, X, y=None):
        """
        Метод для векторизации датасета моделью
            X: pandas.DataFrame/numpy.ndarray
                Датасет (корпус текста)
                
            return: scipy.sparse.csr.csr_matrix
                Возвращает векторизированную разряженную матрицу
        """
        if self.kind_vector == 'tf_idf':
            return self.vectorizer.transform(X)
        elif self.kind_vector == 'word2vec':
            return csr_matrix(np.vstack([self.__word_averaging(review, self.vectorizer.wv) for review in X]))
    
    def fit_transform(self, X, y=None):
        """
        Метод для обучения и векторизации датасета моделью
            X: pandas.DataFrame/numpy.ndarray
                Датасет (корпус текста)
                
            return: scipy.sparse.csr.csr_matrix
                Возвращает векторизированную разряженную матрицу
        """
        self.X_init = X
        
        if self.kind_vector == 'tf_idf':
            self.vectorizer.fit(X)
            return self.vectorizer.transform(X)
        elif self.kind_vector == 'word2vec':
            return csr_matrix(np.vstack([self.__word_averaging(review, self.vectorizer.wv) for review in X]))
        
    def __word_averaging(self, text, wv):
        """
        Метод для усреднения векторов слов в модели word2vec
            text: str
                Документ текста (строка текста)
            wv: object of word vector (example - gensim.models.keyedvectors.Word2VecKeyedVectors)
                Вектор слов
                
            return: numpy.ndarray
                Усредненный вектор слов
        """
        words = text.split(' ')
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not mean:
            print(f"cannot compute similarity with no input {words}")
            # FIXME: remove these examples in pre-processing
            return np.zeros(wv.syn0norm.shape[1],)

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        
        return mean