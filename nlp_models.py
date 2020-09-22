from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
import numpy as np


class ModelsClassical():
    """
    Класс для моделей классического МО
    """
    def __init__(self, type_class='binary', y=None, type_model='lin', **kwargs):
        """
        Выбор модели
            type_class: {'binary', 'multy'}, default='binary'
                Тип классификации - бинарная или многоклассовая
            y: numpy.ndarray, default=None
                Массив целевой переменной. Нобходим для нахождения уникальных классов в многоклассовой классификации.
            type_model: {'lin', 'lgbm', 'svm'}, default='lin'
                Тип модели
            **kwargs
                Гиперпараметры для моделей
        """
        if type_model == 'lin':
            self.model = LogisticRegression(**kwargs)
        elif type_model == 'lgbm':
            self.model = LGBMClassifier(boosting_type='gbdt', n_jobs=20, **kwargs)
        elif type_model == 'svm':
            # Пока заглушка - метод не реализован
            pass

        if type_class != 'binary' and type_model != 'lgbm':
            self.model = OneVsRestClassifier(self.model)
            self.mlb = MultiLabelBinarizer(classes=sorted(np.unique(y)))
        elif type_class != 'binary' and type_model == 'lgbm':
            self.model.set_params(objective='multiclass', metric='multi_logloss')
        
        self.type_class = type_class
        self.type_model = type_model
        self.kwargs = kwargs
        
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """
        Обучение модели
            X_train: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Тренировочный датасет
            X_test: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix, default=None
                Тестовый датасет
            y_train: pandas.Series/numpy.ndarray
                ЦП соответствующая тренироврочному датасету
            y_test: pandas.Series/numpy.ndarray, default=None
                ЦП соответствующая тестовому датасету
        """
        if self.type_class != 'binary' and self.type_model != 'lgbm':
            self.y_train = self.mlb.fit_transform(y_train.apply(lambda x: [x]))
        
        if self.type_model != 'lgbm':
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, verbose=False)
           
    def predict(self, X, threshold=False):
        """
        Прогнозирование класса
            X: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Датасет
            threshold:
                Порог для класса
                
            return: numpy.ndarray
                Массив с предсказанными классами
        """
        if threshold:
            def __proba_to_tag(line, thresh):
                """
                Внутреняя функция для превращения вероятности в соответствие классу согласно порогу.
                Пример: есть три класса со следующими вероятностями [0.58, 0.32, 0.1]. Первый класс всегда others, в который падают обычно все неинтересующие текста. Класс самый большой. Алогритм результирующего класса следующий. Если вероятность первого класса (others) больше остальных, то берется максимальная вероятность из всех классов, кроме первого (others). Если этот максимум больше порога, то объект относится к этому классу, иначе относится к первому классу.
                    line: numpy.ndarray
                        Массив вероятностей каждого класса
                    thresh: float
                        Порог
                        
                    return: numpy.ndarray
                """
                max_line = max(line)
                
                if line[0] == max_line:
                    max_prob = max(line[1:])
                    if max_prob > thresh:
                        index_max = np.where(line==max_prob)[0][0]
                    else:
                        index_max = 0
                else:
                    index_max = np.where(line==max_line)[0][0]
                
                for tag in range(len(line)):
                    if tag == index_max:
                        line[tag] = 1
                    else:
                        line[tag] = 0   
                        
                return line.astype(int)
            
            y_pred_proba = self.model.predict_proba(X)
            if self.type_class == 'binary':
                y_final = (y_pred_proba[:,-1] > threshold)*1
            else:
                y_pred = np.apply_along_axis(__proba_to_tag, axis=1, arr=y_pred_proba, thresh=threshold)
                y_final = self.__transform_multyclass_to_one_list(y_pred)
            return y_final
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Прогнозирование вероятности класса
            X: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Датасет
                
            return: numpy.ndarray
                Массив с вероятностями для каждого класса
        """
        return self.model.predict_proba(X)
    
    def transform_multy_class(self, y):
        """
        Преобразование массива к необходимому для подбора гиперпараметров виду. 
        Кажый объект массива оборачивается в список.
            y: numpy.ndarray
                Целевая переменная
                
            return: numpy.ndarray
                Массив с классами, обернутыми в список для каждого объекта
        """
        if self.type_model != 'lgbm':
            return self.mlb.fit_transform(y.apply(lambda x: [x]))
    
    def __transform_multyclass_to_one_list(self, y):
        """
        Трансформация массива, состоящего из массивов (единица стоит на индексе предсказанного класса), к массиву, состоящему просто из номеров классов.
            y: numpy.ndarray
                Целевая переменная
                
            return: numpy.ndarray
                Массив из номеров классов
        """
        def __transform_one_line(line):
            """
            Внутренняя функция для трансформации одного объекта массива
                line: numpy.ndarray
                    Отдельный масив
                    
                return: int
            """
            if np.max(line) == 0:
                return 0
            else:
                return np.where(line == 1)[0][0]
        
        return np.apply_along_axis(__transform_one_line, 1, y)