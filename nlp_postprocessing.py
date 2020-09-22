from itertools import product
from copy import deepcopy
import pickle
import json
import sys
import os

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, log_loss, confusion_matrix
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import numpy as np
import pandas as pd
import gensim
import sklearn
import lightgbm


class PostProcessing():
    """
    Класс для оценки моделей
    """
    def __init__(self, model=None, X=None, y=None, type_classification=None,
                 train_test_flag=False, X_2=None, y_2=None, average='weighted', threshold=0.5):
        """
        Рассчет стандартных метрик (точность, полнота, матрица неточностей для многоклассовой и бинарной +
        ROC AUC и logloss для бираной) и сведение их в единый словарь
            model: object of model
                Обученная модель МО
            X: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Датасет (в случае train_test_flag=True - тренировочный датасет)
            y: pandas.Series/numpy.ndarray
                Массив целевых переменных (в случае train_test_flag=True - соответствует тренировочному датасету)
            type_classification: {'binary', 'multy'}
                Тип клаассификации - бинарная или многоклассовая
            train_test_flag: bool, default=False
                Флаг для автоматического постпроцессинга для тренировочного и тестового датасетов
            X_2: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
                Тестовый датасет
            y_2: pandas.Series/numpy.ndarray
                ЦП соответствующая тестовому датасету
            average: {'micro', 'macro', 'weighted'}, default='weighted'
                Параметр, необходимый для усредения метрики для многоклассовой классификации
            threshold: float, default=0.5
                Порог для бинарной классификации
        """
        self.dict_results = {}
        if model is not None:
            if type_classification == 'binary':
                self.__get_binary_eval(model, X, y, train_test_flag, X_2, y_2, threshold)
            else:
                self.__get_multy_eval(model, X, y, train_test_flag, X_2, y_2, average)

    def __get_binary_eval(self, model, X, y, train_test_flag, X_2, y_2, threshold):
        """
        Встроенная функция для формирования словаря для бинарной классификации
        """
        y_pred = model.predict(X, threshold=threshold)
        y_pred_proba = model.predict_proba(X)[:, -1]

        if train_test_flag:
            y_pred_2 = model.predict(X_2, threshold=threshold)
            y_pred_proba_2 = model.predict_proba(X_2)[:, -1]

            cm_1 =  confusion_matrix(y, y_pred).tolist()
            cm_2 = confusion_matrix(y_2, y_pred_2).tolist()

            for i in range(len(cm_1)):
                for j in range(len(cm_1[i])):
                    cm_1[i][j] = int(cm_1[i][j])

            for i in range(len(cm_2)):
                for j in range(len(cm_2[i])):
                    cm_2[i][j] = int(cm_2[i][j])

            self.dict_results['confusion_matrix'] = {'train': cm_1,
                                                     'test': cm_2}
            self.dict_results['f1'] = {'train': round(f1_score(y, y_pred), 5),
                                       'test': round(f1_score(y_2, y_pred_2), 5)}
            self.dict_results['precision'] = {'train': round(precision_score(y, y_pred), 5),
                                              'test': round(precision_score(y_2, y_pred_2), 5)}
            self.dict_results['recall'] = {'train': round(recall_score(y, y_pred), 5),
                                           'test': round(recall_score(y_2, y_pred_2), 5)}
            self.dict_results['roc_auc'] = {'train': round(roc_auc_score(y, y_pred_proba), 5),
                                            'test': round(roc_auc_score(y_2, y_pred_proba_2), 5)}
            self.dict_results['log_loss'] = {'train': round(log_loss(y, y_pred_proba), 5),
                                             'test': round(log_loss(y_2, y_pred_proba_2), 5)}
        else:
            cm = confusion_matrix(y, y_pred).tolist()
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    cm[i][j] = int(cm[i][j])

            self.dict_results['confusion_matrix'] = cm
            self.dict_results['f1'] = round(f1_score(y, y_pred), 5)
            self.dict_results['precision'] = round(precision_score(y, y_pred), 5)
            self.dict_results['recall'] = round(recall_score(y, y_pred), 5)
            self.dict_results['roc_auc'] = round(roc_auc_score(y, y_pred_proba), 5)
            self.dict_results['log_loss'] = round(log_loss(y, y_pred_proba), 5)

    def __get_multy_eval(self, model, X, y, train_test_flag, X_2, y_2, average):
        """
        Встроенная функция для формирования словаря для многоклассовой классификации
        """
        y_pred = model.predict(X)
        my_tags = sorted(y.unique())

        if train_test_flag:
            y_pred_2 = model.predict(X_2)
            cm_1 = confusion_matrix(y, y_pred, labels=my_tags)
            cm_2 = confusion_matrix(y_2, y_pred_2, labels=my_tags)

            cm_1_log = cm_1.tolist()
            cm_2_log = cm_2.tolist()
            for i in range(len(cm_1_log)):
                for j in range(len(cm_1_log[i])):
                    cm_1_log[i][j] = int(cm_1_log[i][j])

            for i in range(len(cm_2_log)):
                for j in range(len(cm_2_log[i])):
                    cm_2_log[i][j] = int(cm_2_log[i][j])

            self.dict_results['confusion_matrix'] = {'train': cm_1_log,
                                                     'test': cm_2_log}
            self.dict_results[f'f1_{average}'] = {'train': round(f1_score(y, y_pred, average=average), 5),
                                                  'test': round(f1_score(y_2, y_pred_2, average=average), 5)}
            self.dict_results[f'precision_{average}'] = {'train': round(precision_score(y, y_pred, average=average),
                                                                        5),
                                                         'test': round(precision_score(y_2, y_pred_2, average=average),
                                                                       5)},
            self.dict_results[f'recall_{average}'] = {'train': round(recall_score(y, y_pred, average=average), 5),
                                                      'test': round(recall_score(y_2, y_pred_2, average=average), 5)}

            clas = {}

            for i in my_tags:
                clas[f'{i}'] = {}
                prec_train = round(cm_1[i][i] / sum(cm_1.T[i]), 5)
                prec_test = round(cm_2[i][i] / sum(cm_2.T[i]), 5)
                rec_train = round(cm_1[i][i] / sum(cm_1[i]), 5)
                rec_test = round(cm_2[i][i] / sum(cm_2[i]), 5)
                clas[f'{i}']['precision'] = {'train': prec_train,
                                             'test': prec_test}
                clas[f'{i}']['recall'] = {'train': rec_train,
                                          'test': rec_test}
                clas[f'{i}']['f1'] = {'train': round((2 * prec_train * rec_train) / (prec_train + rec_train), 5),
                                      'test': round((2 * prec_test * rec_test) / (prec_test + rec_test), 5)}


            self.dict_results['metrics_by_class'] = clas

        else:
            cm = confusion_matrix(y, y_pred, labels=my_tags)
            cm_log = cm.tolist()
            for i in range(len(cm_log)):
                for j in range(len(cm_log[i])):
                    cm_log[i][j] = int(cm_log[i][j])

            self.dict_results['confusion_matrix'] = cm_log
            self.dict_results[f'f1_{average}'] = round(f1_score(y, y_pred, average=average), 5)
            self.dict_results[f'precision_{average}'] = round(precision_score(y, y_pred, average=average), 5)
            self.dict_results[f'recall_{average}'] = round(recall_score(y, y_pred, average=average), 5)

            clas = {}

            for tag in my_tags:
                clas[f'{tag}'] = {}
                prec = round(cm[i][i] / sum(cm.T[i]), 5)
                rec = round(cm[i][i] / sum(cm[i]), 5)
                clas[f'{tag}']['precision'] = prec
                clas[f'{tag}']['recall'] = rec
                clas[f'{tag}']['f1'] = round((2 * prec * rec) / (prec + rec), 5)

            self.dict_results['metrics_by_class'] = clas

    def evaluate_train_test(self, y_train_pred, y_train, y_test_pred, y_test,
                            tag=None, aver='weighted', dict_classes=None):
        """
        Функция для красивого представления полноты табличкой для каждого класса (для треина и теста)
        + табличка с агрегированными точностью, полнотой и F-мерой для треина и теста
        + матрицы неточностей для треина и теста
            y_train_pred: pandas.Series/numpy.ndarray
                Предсказание для тренировочного датасета
            y_train: pandas.Series/numpy.ndarray
                ЦП соответствующее тренировочному датасету
            y_test_pred: pandas.Series/numpy.ndarray
                Предсказание для тестового датасета
            y_test: pandas.Series/numpy.ndarray
                ЦП соответствующее тестовому датасету
            aver: {'micro', 'macro', 'weighted'}, default='weighted'
                Параметр, необходимый для усредения метрики для многоклассовой классификации
            dict_classes: dict, default=None
                Словарь для расшифровки классов
        """
        if dict_classes is None:
            dict_classes = {}
            for i in y_train.unique():
                dict_classes[i] = i

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        fig.tight_layout()

        ax_im_0, im_0, cm_0 = self.__evaluate_prediction(axes[0], y_train_pred, y_train,
                                                  title='Train')
        fig.colorbar(im_0, ax=ax_im_0)

        ax_im_1, im_1, cm_1 = self.__evaluate_prediction(axes[1], y_test_pred, y_test,
                                                  title='Test')
        fig.colorbar(im_1, ax=ax_im_1) 

        table_only_recall = PrettyTable(['', 'Train Recall', 'Test Recall'])

        for i in range(len(cm_0)):
            if tag is not None and i != 0:
                index_tag = dict_classes[int(tag)]
            else:
                index_tag = dict_classes[i]
            table_only_recall.add_row([index_tag,
                                       round(cm_0[i][i] / sum(cm_0[i]), 5),
                                       round(cm_1[i][i] / sum(cm_1[i]), 5)])

        table_train_test = PrettyTable(['', 'Train', 'Test'])

        table_train_test.add_row([f'recall {aver}', 
                       round(recall_score(y_train, y_train_pred, average=aver), 5), 
                       round(recall_score(y_test, y_test_pred, average=aver), 5)])
        table_train_test.add_row([f'precision {aver}', 
                       round(precision_score(y_train, y_train_pred, average=aver), 5), 
                       round(precision_score(y_test, y_test_pred, average=aver), 5)])
        table_train_test.add_row([f'f1 {aver}', 
                       round(f1_score(y_train, y_train_pred, average=aver), 5), 
                       round(f1_score(y_test, y_test_pred, average=aver), 5)])

        print(table_only_recall)
        print(table_train_test)
        
    def __evaluate_prediction(self, ax, y_pred, y, title="Confusion matrix", normalize=True):
        """
        Внутренняя функция для построения матрицы неточностей (вместе с картинками)
        """
        my_tags = sorted(y.unique())
        cm = confusion_matrix(y, y_pred, labels=my_tags)

        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)

        ax.set_title(title, fontsize=20)

        ax.set_xticks(my_tags)
        ax.set_yticks(my_tags)

        ax.set_ylabel('True label', fontsize=12)
        ax.set_xlabel('Predicted label', fontsize=12)

        if normalize:
            cm_fin = cm_normalized
            fmt = '.2f'
        else:
            cm_fin = cm
            fmt = '2d'

        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], '2d'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        return ax, im, cm

    def most_influential_words(self, vectorizer, model, tag_class=0, num_words=20):
        """
        Функция для подсчета самых часто встречающихся слов для каждого класса (пока только для бинарной классификации)
            vectorizer: object of vectorizer
                Объект класса векторизации
            model: object of model
                Объект класса модели
            genre_index: int, default=0
                Тег класса
            num_words: int, default=20
                Кол-во выводимых слов

            return: list
                Список самых часто встречающихся слов длинной num_words
        """
        if type(vectorizer) == gensim.models.keyedvectors.Word2VecKeyedVectors:
            features = vectorizer.get_feature_names()
        elif type(vectorizer) == sklearn.feature_extraction.text.TfidfVectorizer:
            tfidf_vocab = vectorizer.vocabulary_
            features = {i: word for word, i in tfidf_vocab.items()}

        if type(model) == lightgbm.sklearn.LGBMClassifier:
            max_coef = sorted(enumerate(model.feature_importances_), key=lambda x: x[1], reverse=True)
        elif type(model) == sklearn.linear_model._logistic.LogisticRegression:
            max_coef = sorted(enumerate(model.coef_[0]), key=lambda x: x[1], reverse=True)

        return [features[x[0]] for x in max_coef[:num_words]]

        # if type(vectorizer) == gensim.models.keyedvectors.Word2VecKeyedVectors:
        #     features = vectorizer.get_feature_names()
        #     max_coef = sorted(enumerate(model.coef_[tag_class]), key=lambda x: x[1], reverse=True)
        #     return [features[x[0]] for x in max_coef[:num_words]]
        # elif type(vectorizer) == sklearn.feature_extraction.text.TfidfVectorizer:
        #     tfidf_vocab = vectorizer.vocabulary_
        #     tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}
        #     if type(model) == lightgbm.sklearn.LGBMClassifier:
        #         feat = sorted(enumerate(model.feature_importances_), key=lambda x: x[1], reverse=True)
        #     elif type(model) == sklearn.linear_model._logistic.LogisticRegression:
        #         feat = sorted(enumerate(model.model.coef_[0]), key=lambda x: x[1], reverse=True)
        #     return [tfidf_reversed_vocab[feat[i][0]] for i in range(num_words)]


    def comparison(self, y_pred, y, indexes,
                   column_text, column_target, target,
                   data_init=None, way_to_data=None):
        """
        Функция для просмотра неверно классифицированных текстов
            y_pred: pandas.Series/numpy.ndarray
                Массив предсказанных классов
            y: pandas.Series/numpy.ndarray
                Массив реальных ЦП
            indexes: list/numpy.ndarray
                Массив изначальных индексов
            column_text: str
                Название колонки с текстом
            column_target: str
                Название колонки с целевой переменной
            target: int
                Класс для которого хочется посмотреть неверно классифицированные тексты
            data_init: pandas.DataFrame, default=None
                Изначальный датасет с текстами
            way_to_data: str, default=None
                Путь до изначального датасета с текстами

            return: pandas.DataFrame
                Датафреим с неверно классифицированными текстами
        """
        data = pd.DataFrame(y.values, columns=(column_target,), index=indexes)
        data['pred'] = y_pred

        indexes_y = data[(data[column_target] == target) &
                         (data[column_target] != data['pred'])].index

        if data_init is None:
            way = way_to_data[:way_to_data.find(')/')+2]
            data_init = pd.read_pickle(f'{way}data_init_run.pickle')

        return data_init.loc[indexes_y][[column_text, column_target]]

    def get_best_experiment_for_class(self, evaluation, init_folder='results', limit_eval=0.0, tag='1',
                                      threshold_eval='precision', threshold_value=0.0,
                                      threshold_diff=0.1, numb=0):
        """
        Функция для получения самого лучшего эксперимента из пула экспериментов
            evaluation: str
                Оценка по которой надо искать лучший эксперимент
            init_folder: str, default='results'
                Исходная папка
            limit_eval: float, default=0.0
                Значение порога для оценки (ниже порога результаты не рассматриваются)
            tag: str, default= '1'
                Класс по которому ищутся результаты
            threshold_eval: str, default='precision'
                Дополнительная ограничивающая оценка
            threshold_value: float, default=0.0
                Значение порого для дополнительной ограничивающей оценки
            threshold_diff: float, default=0.1
                Значение порога для разницы по метрике между треином и тестом
            numb: int, default=0
                Значение порядкового номера эксперимента в отсортированном по максимизированной метрике массиве

            return: dict, str, float
                Словарь с результатами и исходными данными эксперимента, относительный путь до этого словаря
                и порог
        """
        find_files = []
        for root, dirs, files in os.walk(f'{init_folder}/'):
            find_files += [os.path.join(root, name) for name in files if name[-23:] == 'experiment_results.json']

        data_results = pd.DataFrame(columns=(evaluation, 'diff_train_test', f'{evaluation}_div_diff_train_test'))

        for file in find_files:
            # if file.find('binary') != -1 and file[len(init_folder)+12] != tag:  #прод - 19, тест - 24
            if file.find('binary') != -1 and file.find(f'(0, {tag})/') == -1:  #прод - 19, тест - 24  file[28] != tag
                continue

            with open(file, 'r') as f:
                results = json.load(f)

            if results['launch_parameters']['classification_type'] == 'binary':
                result_eval = results['evaluation'][evaluation]['test']
                diff_train_test_eval = results['evaluation'][evaluation]['train'] \
                                       - results['evaluation'][evaluation]['test']
                eval_by_threshold = results['evaluation'][threshold_eval]['test']
            else:
                result_eval = results['evaluation']['metrics_by_class'][tag][evaluation]['test']
                diff_train_test_eval = results['evaluation']['metrics_by_class'][tag][evaluation]['train'] \
                                       - results['evaluation']['metrics_by_class'][tag][evaluation]['test']
                eval_by_threshold = results['evaluation']['metrics_by_class'][tag][threshold_eval]['test']

            if (eval_by_threshold >= threshold_value) and (result_eval >= limit_eval):
                try:
                    div = result_eval / abs(diff_train_test_eval)
                except ZeroDivisionError:
                    div = result_eval
                data_results.loc[int(results['experiment_number'])] = [result_eval, diff_train_test_eval, div]

        try:
            # data_results = data_results.sort_values(by=f'{evaluation}_div_diff_train_test', ascending=False).reset_index()
            data_results = data_results.sort_values(by=f'{evaluation}', ascending=False).reset_index()
            data_results = data_results[data_results['diff_train_test'] < threshold_diff]
        except TypeError:
            print(data_results)

        try:
            best_experiment = int(data_results.iloc[numb]["index"])
        except IndexError:
            print('Таких экспериментов нет!')
            return None, None, None

        print(f'Лучший результат был показан для эксперимента № {best_experiment}:\n')
        print(f'{evaluation} - {data_results.iloc[numb][evaluation]}')
        print(f'Разница {evaluation} между треином и тестом - {data_results.iloc[numb]["diff_train_test"]}')

        for file in find_files:
            if file.find(f'/{str(int(data_results.iloc[numb]["index"]))}_') != -1:
                with open(file, 'r') as f:
                    final_results, final_way = json.load(f), file

        print(f'Порог - {final_results["launch_parameters"]["models"]["threshold"]}')

        return final_results, final_way, float(final_results["launch_parameters"]["models"]["threshold"])

    def get_part_of_experiment(self,  way_to_json, data_prepare_flag=False, vect_flag=False, data_vect_flag=False,
                               split_flag=False, x_train_flag=False, x_test_flag=False,
                               y_train_flag=False, y_test_flag=False,
                               train_indexes_flag=False, test_indexes_flag=False, model_flag=False):
        """
        Функция для получения частей эксперимента (предобработанного датасета, треина и теста, разных моделей)
            way_to_json: str
                Путь до json файла с результатами
            data_prepare_flag: bool, default=False
                Флаг добавления в итоговый словарь предобработанного датасета
            vect_flag: bool, default=False
                Флаг добавления в итоговый словарь модели по векторизации
            data_vect_flag: bool, default=False
                Флаг добавления в итоговый словарь векторизированного датасета
            split_flag: bool, default=False
                Флаг добавления в итоговый словарь модели разделения на треин и тест
            x_train_flag: bool, default=False
                Флаг добавления в итоговый словарь тренировочного датасета
            x_test_flag: bool, default=False
                Флаг добавления в итоговый словарь тестового датасета
            y_train_flag: bool, default=False
                Флаг добавления в итоговый словарь массива ЦП соответстуюих тренировочному датасету
            y_test_flag: bool, default=False
                Флаг добавления в итоговый словарь массива ЦП соответстуюих тестовому датасету
            model_flag:bool, default=False
                Флаг добавления в итоговый словарь обученной модели машинного обучения

            return: dict
                Словарь с необходимыми элементами (в случае флага False для элемента значение ключа равно None)
        """

        def get_way(way_all, part_flag):
            """
            Внутренняя функция получения пути до элемента
            """
            way = ''
            for part in way_all.split('/'):
                if part_flag == 'model':
                    if (part.find('lin') != -1) or (part.find('lgbm') != -1):
                        break
                elif part_flag == 'json':
                    if part.find('json') != -1:
                        break
                else:
                    if part.find(part_flag) != -1:
                        break
                way += f'{part}/'
            return way

        if data_prepare_flag:
            data_prepare = pd.read_pickle(f"{get_way(way_to_json, 'vectorization')}X_prepare.pickle")
        else:
            data_prepare = None
        if vect_flag:
            with open(f"{get_way(way_to_json, 'train_test')}vectorization.pickle", 'rb') as file:
                vect_model = pickle.load(file)
        else:
            vect_model = None
        if data_vect_flag:
            with open(f"{get_way(way_to_json, 'train_test')}X_vect.pickle", 'rb') as file:
                data_vect = pickle.load(file)
        else:
            data_vect = None
        if split_flag:
            with open(f"{get_way(way_to_json, 'model')}split.pickle", 'rb') as file:
                split_model = pickle.load(file)
        else:
            split_model = None
        if x_train_flag:
            with open(f"{get_way(way_to_json, 'model')}X_train.pickle", 'rb') as file:
                X_train = pickle.load(file)
        else:
            X_train = None
        if x_test_flag:
            with open(f"{get_way(way_to_json, 'model')}X_test.pickle", 'rb') as file:
                X_test = pickle.load(file)
        else:
            X_test = None
        if y_train_flag:
            with open(f"{get_way(way_to_json, 'model')}y_train.pickle", 'rb') as file:
                y_train = pickle.load(file)
        else:
            y_train = None
        if y_test_flag:
            with open(f"{get_way(way_to_json, 'model')}y_test.pickle", 'rb') as file:
                y_test = pickle.load(file)
        else:
            y_test = None
        if train_indexes_flag:
            with open(f"{get_way(way_to_json, 'model')}train_indexes.pickle", 'rb') as file:
                train_indexes = pickle.load(file)
        else:
            train_indexes = None
        if test_indexes_flag:
            with open(f"{get_way(way_to_json, 'model')}test_indexes.pickle", 'rb') as file:
                test_indexes = pickle.load(file)
        else:
            test_indexes = None
        if model_flag:
            with open(f"{get_way(way_to_json, 'json')}model.pickle", 'rb') as file:
                model = pickle.load(file)
        else:
            model = None

        return {'X_prepare': data_prepare,
                'vectorization_model': vect_model,
                'X_vect': data_vect,
                'split_model': split_model,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'train_indexes': train_indexes,
                'test_indexes': test_indexes,
                'model': model}