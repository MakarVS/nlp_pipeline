from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import pairwise
from scipy.sparse import vstack
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class TrainTestSplit():
    """
    Класс для разделения на треин и тест выборки
    """
    def __init__(self, kind_split='simple', test_size=0.2, random_state=42, 
                 clusters=None, cluster_type_model='mini', **kwargs):
        """
        Выбор разделения
            kind_splint: {'simple', 'something', 'cluster'}, default='simple'
                Вид разделения 
                'simple' - обычное из sklearn;
                'something' - разделение по дополнительному признаку (к примеру по месяцам года, из каждого месяца берется равное кол-во в треин и тест);
                'cluster' - кластеризация и последующее разбиение по получившимся кластерам
            test_size: float, default=0.2
                Процент тестовой выборки
            random_state: int, default=42
                Управление перемешиванием данных перед разделением. Фиксация воспроизводимого датасета
            clusters: list/numpy.ndarray, default=None
                Список кластеров (или дополнительного признака) по которым надо разделять
            cluster_type_model: {'kmeans', 'mini', 'dbscan', 'aggl'}, default='mini'    
                Тип алгоритма кластеризации в случае, когда kind_splint='cluster'
            **kwargs
                Гиперпараметры для алгоритмов кластеризации
        """
        if kind_split == 'cluster':
            if cluster_type_model == 'kmeans':
                self.cluster_model = KMeans(random_state=random_state, **kwargs)
            elif cluster_type_model == 'dbscan':
                self.cluster_model = DBSCAN(**kwargs)
            elif cluster_type_model == 'mini':
                self.cluster_model = MiniBatchKMeans(random_state=random_state, **kwargs)
            elif cluster_type_model == 'aggl':
                self.cluster_model = AgglomerativeClustering(**kwargs)
        
        self.kind_split = kind_split
        self.test_size = test_size
        self.random_state = random_state
        self.clusters = clusters
        self.cluster_type_model = cluster_type_model
        
    def fit(self, X, y=None):
        """
        В случаях когда kind_split равен 'split' или 'month' ничего не возвращает, нужен для использования объектов класса в пайплайнах sklearn. В случае kind_split='cluster' обучает модель кластеризации.
            X: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Датасет
        """
        self.X_init = X
        self.y_init = y
        
        if self.kind_split == 'simple':
            return self
        elif self.kind_split == 'something':
            return self
        elif self.kind_split == 'cluster':
            if self.cluster_type_model == 'aggl':
                self.cluster_model.fit(X.toarray())
            else:
                self.cluster_model.fit(X)
        
    def transform(self, X, y):
        """
        Разделяет выборку и ЦП на треин и тест согласно выбранному типу разделения
            X: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Датасет
            y: pandas.Series/numpy.ndarray
                Целевая переменная
                            
            return: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
                Возвращает тренировочный датасет, тестовый датасет, ЦП соответствующую тренироврочному датасету, ЦП соответствующую тестовому датасету            
        """
        if self.kind_split == 'simple':
            # X_train, X_test, \
            # y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            # return X_train, X_test, y_train, y_test, X_train.indexes
            return self.__manual_split(X, y, y)
        elif self.kind_split == 'something':
            return self.__manual_split(X, y, self.clusters)
        elif self.kind_split == 'cluster':
            clusters = self.cluster_model.labels_
            return self.__manual_split(X, y, clusters)
        
    def fit_transform(self, X, y):
        """
        Разделяет выборку и ЦП на треин и тест согласно выбранному типу разделения (обучая сразу модели кластеризации)
            X: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Датасет
            y: pandas.Series/numpy.ndarray
                Целевая переменная
                
            return: numpy.ndarray, numpy.ndarray, pandas.Series, pandas.Series
                Возвращает тренировочный датасет, тестовый датасет, ЦП соответствующую тренироврочному датасету, ЦП соответствующую тестовому датасету            
        """
        self.X_init = X
        self.y_init = y
        
        if self.kind_split == 'simple':
            # return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            return self.__manual_split(X, y, y)
        elif self.kind_split == 'something':
            return self.__manual_split(X, y, self.clusters)
        elif self.kind_split == 'cluster':
            if self.cluster_type_model == 'aggl':
                self.cluster_model.fit(X.toarray())
            else:
                self.cluster_model.fit(X)
            clusters = self.cluster_model.labels_
            return self.__manual_split(X, y, clusters)
        
    def viz_cluster(self, X, title=''):
        """
        Визуализация кластеров
            X: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Датасет
            title: str, default=''
                Заголовок графика распределения
        """
        dist = 1 - pairwise.cosine_similarity(X)       
        
        icpa = IncrementalPCA(n_components=2, batch_size=16)
        icpa.fit(dist)
        demo = icpa.transform(dist)
        xs, ys = demo[:, 0], demo[:, 1]
        
        labels = self.cluster_model.labels_
        
        labels_unique = list(set(labels))
        traces = []
        for label in labels_unique:
            indexes = np.where(labels == label)[0]
            trace = {'type': 'scatter',
                     'x': xs[indexes],
                     'y': ys[indexes],
                     'name': int(label),
                     'mode': 'markers',
                     'marker': {'size': 7},
                     'text': X.toarray()[indexes]
                    }
            traces.append(trace)
        layout = go.Layout(title=title, showlegend=True)

        data = go.Data(traces)
        fig = go.Figure(data=data, layout=layout)
        fig.show()
            
    def __manual_split(self, X, y, clusters):
        """
        Собственная реализация разделения по какому-то кластеру/признаку
            X: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Датасет
            y: pandas.Series/numpy.ndarray
                Целевая переменная
            clusters: list/numpy.ndarray, default=None
                Список кластеров (или дополнительного признака) по которым надо разделять
                
            return: numpy.ndarray, numpy.ndarray, pandas.Series, pandas.Series
                Возвращает тренировочный датасет, тестовый датасет, ЦП соответствующую тренироврочному датасету, ЦП соответствующую тестовому датасету   
        """
        np.random.seed(self.random_state)

        list_clusters = list(set(clusters))
        flag = False
        general_train_ind_list, general_test_ind_list = [], []

        for cluster in list_clusters:
            index = np.where(clusters == cluster)[0]
            shuffled_indices = np.random.permutation(index)
            test_set_size = int(len(index) * self.test_size)
            train_indices = shuffled_indices[test_set_size:]
            test_indices = shuffled_indices[:test_set_size]

            general_train_ind_list.extend(y.iloc[train_indices].index)
            general_test_ind_list.extend(y.iloc[test_indices].index)

            if flag:
                X_train = vstack([X_train, X[train_indices]])
                X_test = vstack([X_test, X[test_indices]])
                y_train = np.concatenate([y_train, y.to_numpy()[train_indices]])
                y_test = np.concatenate([y_test, y.to_numpy()[test_indices]])
            else:
                X_train = X[train_indices]
                X_test = X[test_indices]
                y_train = y.to_numpy()[train_indices]
                y_test = y.to_numpy()[test_indices]

            flag = True

        general_train_ind_array = np.array(general_train_ind_list)
        general_test_ind_array = np.array(general_test_ind_list)
        
        shuffled_indices_train = np.random.permutation(range(len(y_train)))
        shuffled_indices_test = np.random.permutation(range(len(y_test)))

        return X_train[shuffled_indices_train], X_test[shuffled_indices_test], \
               pd.Series(y_train[shuffled_indices_train]), pd.Series(y_test[shuffled_indices_test]), \
               general_train_ind_array[shuffled_indices_train], general_test_ind_array[shuffled_indices_test]
