from copy import deepcopy
import pickle
import json
import os

import pandas as pd
import numpy as np

from nlp_pipeline.nlp_preprocessing import choose_target, TextPrepare
from nlp_pipeline.nlp_vectorization import Vectorization
from nlp_pipeline.nlp_train_test_split import TrainTestSplit
from nlp_pipeline.nlp_models import ModelsClassical
from nlp_pipeline.nlp_search_hyperparameters import search_hyper
from nlp_pipeline.nlp_postprocessing import PostProcessing


class RunNLPPipeleine():
    """
    Класс для запуска всего пайпалайна с сохранением результатов
    """
    def __init__(self, data, column_text, column_target, launch_param,
                 init_folder='results', lang='ru', number=None):
        """
        Сохранение атрибутов для запуска
            data: pandas.DataFrame
                Датафреим
            column_text: str
                Название колонки с текстом
            column_target: str
                Название колонки с целевой переменной
            launch_param: dict
                Словарь с параметрами запуска.
                Ключи в словаре:
                classification_type: {'multy', 'binary'}
                    Тип классификации - многоклассовая или бинарная
                selected_targets: tuple, default=None
                    Выбранные классы для бинарной классификации
                text_prepare: dict
                    Словарь с параметрами предварительной обработки текста
                    (параметры см. в nlp_preprocessing класс TextPrepare)
                vectorization: dict
                    Словарь с параметрами векторизации текста (параметры см. в nlp_vectorization класс Vectorization)
                train_test_split: dict
                    Словарь с параметрами разделения на треин и тест
                    (параметры см. в nlp_train_test_split класс TrainTestSplit)
                models: dict
                    Словарь с параметрами модели и параметрами, требующимеся для оценки модели
                    (параметры см. в nlp_models класс ModelsClassical, в nlp_search_hyperparameters функцию search_hyper
                    и в nlp_postprocessing класс PostProcessing)
            lang: {'ru', 'eng'}, default='ru'
                Язык текста ПОКА ТОЛЬКО RUS
            init_folder: str, default='results'
                Папка, в которую будут записываться результаты
            number: int, default=None
                Номер эксперимента
        """
        self.data = data
        self.column_text = column_text
        self.column_target = column_target
        self.launch_param = launch_param
        self.lang = lang

        self.current_way = f'{init_folder}/'

        if lang == 'ru':
            with open('nlp_pipeline/stopwords_ru.txt') as file:
                self.stopwords = file.readline().split(', ')
        elif lang == 'eng':
            print('Пока что английский текст не реализован!')

        self.log = {'experiment_number': str(number),
                    'launch_parameters': deepcopy(launch_param)}

        self.log['launch_parameters']['train_test_split'].pop('clusters')
        self.log['launch_parameters']['models']['func_eval'] = launch_param['models']['func_eval'].__name__

        self.log['launch_parameters']['vectorization'].pop('word_2_vec_model')
        self.log['launch_parameters']['models']['y'] = np.unique(launch_param['models']['y']).tolist()

        
    def run(self):
        """
        Запуск пайплайна
        """
        if not os.path.exists(f"{self.current_way}{self.launch_param['classification_type']}"):
            os.mkdir(f"{self.current_way}{self.launch_param['classification_type']}")
        self.current_way += self.launch_param['classification_type'] + '/'
                     
        if self.launch_param['classification_type'] == 'binary':
             self.data = choose_target(self.data, target=self.launch_param['selected_targets'])

             if sorted(self.data[self.column_target].unique())[1] != 1:
                 self.data[self.column_target][self.data[self.column_target] != 0] = 1
             if self.launch_param['train_test_split']['kind_split'] == 'something':
                 clust = self.launch_param['train_test_split']['clusters']
                 self.launch_param['train_test_split']['clusters'] = clust[np.isin(clust.index, self.data.index)]
             if not os.path.exists(f"{self.current_way}{self.launch_param['selected_targets']}"):
                 os.mkdir(f"{self.current_way}{self.launch_param['selected_targets']}")
             self.current_way += str(self.launch_param['selected_targets']) + '/'

        # self.data = self.data.reset_index(drop=True)

        self.data.to_pickle(f'{self.current_way}data_init_run.pickle')

        self.func_text_prepare()
        self.func_text_vectorization()
        self.func_train_test_split()
        self.func_models()
        self.func_eval()
        self.logging()

        print(f'Эксперимент {self.log["experiment_number"]} поставлен!')
        
    def func_text_prepare(self):
        """
        Предобработка текста
        """
        self.type_preproc = self.launch_param['text_prepare']['type_preproc']

        preproc_hyper = ''
        if self.type_preproc == 'lemm':
            preproc_hyper += f"/speech_2_vec_lemm_{self.launch_param['text_prepare']['speech_2_vec_lemm']}"
        elif self.type_preproc == 'udp':
            preproc_hyper += f"/keep_pos_udp_{self.launch_param['text_prepare']['keep_pos_udp']}_keep_punct_udp_{self.launch_param['text_prepare']['keep_punct_udp']}"

        way = f'{self.current_way}prepare_{self.type_preproc}{preproc_hyper}'
        if os.path.exists(f'{way}/X_prepare.pickle'):
            self.X_prepare = pd.read_pickle(f'{way}/X_prepare.pickle')
            way_hyp = way
        else:
            self.text_prepare = TextPrepare(self.stopwords, 
                                            type_preproc=self.launch_param['text_prepare']['type_preproc'], 
                                            speech_2_vec_lemm=self.launch_param['text_prepare']['speech_2_vec_lemm'],
                                            way_to_udp=self.launch_param['text_prepare']['way_to_udp'],
                                            keep_pos_udp=self.launch_param['text_prepare']['keep_pos_udp'],
                                            keep_punct_udp=self.launch_param['text_prepare']['keep_punct_udp'],
                                            parallel_workers=self.launch_param['text_prepare']['parallel_workers'])

            self.X_prepare = self.text_prepare.fit_transform(self.data[self.column_text])

            way = f'{self.current_way}prepare_{self.type_preproc}'
            way_hyp = way + f'/{preproc_hyper}'

            if os.path.exists(way):
                if not os.path.exists(way_hyp):
                    os.mkdir(way_hyp)
            else:
                os.mkdir(way)
                if f'{way}/' != way_hyp:
                    os.mkdir(way_hyp)

            # os.mkdir(f'{self.current_way}prepare_{self.type_preproc}')
            # self.X_prepare.to_pickle(f'{self.current_way}prepare_{self.type_preproc}/X_prepare.pickle')

            self.X_prepare.to_pickle(f'{way_hyp}/X_prepare.pickle')
            # print(f'Запись предобработанного текста - {way_hyp}')

        # self.current_way += f'prepare_{self.type_preproc}/'
        self.current_way = f'{way_hyp}/'

    def func_text_vectorization(self):
        """
        Векторизация текста
        """
        self.type_vector = self.launch_param['vectorization']['kind_vector']

        self.vector_hyper = ''
        if self.launch_param['vectorization']['params'] is not None:
            self.vector_hyper += '/'
            for par, val in self.launch_param['vectorization']['params'].items():
                self.vector_hyper += str(par) + '_' + str(val) + '_'
        
        way = f'{self.current_way}vectorization_{self.type_vector}{self.vector_hyper}'
        if os.path.exists(f'{way}/X_vect.pickle'):
            self.X_vect = pd.read_pickle(f'{way}/X_vect.pickle')
            with open(f'{way}/vectorization.pickle', 'rb') as file:
                self.vectorization = pickle.load(file)
        else:
            if self.launch_param['vectorization']['params'] is not None:
                self.vectorization = Vectorization(kind_vector=self.launch_param['vectorization']['kind_vector'],
                                                word_2_vec_model=self.launch_param['vectorization']['word_2_vec_model'],
                                                **self.launch_param['vectorization']['params'])    
            else:
                self.vectorization = Vectorization(kind_vector=self.launch_param['vectorization']['kind_vector'],
                                                word_2_vec_model=self.launch_param['vectorization']['word_2_vec_model'])

            self.X_vect = self.vectorization.fit_transform(self.X_prepare)
            
            way = f'{self.current_way}vectorization_{self.type_vector}'
            way_hyp = way + f'/{self.vector_hyper}'
            
            if os.path.exists(way):
                if not os.path.exists(way_hyp):
                    os.mkdir(way_hyp)  
            else:
                os.mkdir(way)
                if f'{way}/' != way_hyp:
                    os.mkdir(way_hyp)

            with open(f'{way_hyp}/X_vect.pickle', 'wb') as file:
                pickle.dump(self.X_vect, file)

            with open(f'{way_hyp}/vectorization.pickle', 'wb') as file:
                pickle.dump(self.vectorization, file)

            # print(f'Запись векторизированного текста - {way_hyp}')
                
        self.current_way += f'vectorization_{self.type_vector}{self.vector_hyper}/'


    def func_train_test_split(self):
        """
        Разделение на тренировочную и тестовую выборку
        """
        self.type_split = self.launch_param['train_test_split']['kind_split']
        
        self.split_hyper = ''
        
        if self.launch_param['train_test_split']['params'] is not None:
            self.split_hyper += '/'
            for par, val in self.launch_param['train_test_split']['params'].items():
                self.split_hyper += str(par) + '_' + str(val) + '_'
        
        way = f'{self.current_way}train_test_split_{self.type_split}'
        
        if self.type_split == 'cluster': 
            way += f"/{self.launch_param['train_test_split']['cluster_type_model']}"
            
        way += self.split_hyper

        if os.path.exists(f'{way}/X_train.pickle'):
            with open(f'{way}/X_train.pickle', 'rb') as file:
                self.X_train = pickle.load(file)
            with open(f'{way}/X_test.pickle', 'rb') as file:
                self.X_test = pickle.load(file)
            with open(f'{way}/y_train.pickle', 'rb') as file:
                self.y_train = pickle.load(file)
            with open(f'{way}/y_test.pickle', 'rb') as file:
                self.y_test = pickle.load(file)
            with open(f'{way}/split.pickle', 'rb') as file:
                self.split = pickle.load(file)

            self.current_way = way
        else:
            if self.launch_param['train_test_split']['params'] is not None:
                self.split = TrainTestSplit(kind_split=self.launch_param['train_test_split']['kind_split'],
                                        test_size=self.launch_param['train_test_split']['test_size'],
                                        random_state=self.launch_param['train_test_split']['random_state'],
                                        clusters=self.launch_param['train_test_split']['clusters'],
                                        cluster_type_model=self.launch_param['train_test_split']['cluster_type_model'],
                                        **self.launch_param['train_test_split']['params'])
            else:
                self.split = TrainTestSplit(kind_split=self.launch_param['train_test_split']['kind_split'],
                                        test_size=self.launch_param['train_test_split']['test_size'],
                                        random_state=self.launch_param['train_test_split']['random_state'],
                                        clusters=self.launch_param['train_test_split']['clusters'],
                                        cluster_type_model=self.launch_param['train_test_split']['cluster_type_model'])
            
            self.X_train, self.X_test, \
            self.y_train, self.y_test, \
            self.train_ind, self.test_ind = self.split.fit_transform(self.X_vect, self.data[self.column_target])
            
            way = f'{self.current_way}train_test_split_{self.type_split}/'
            self.current_way += f'train_test_split_{self.type_split}/'
            
            if self.type_split == 'cluster': 
                way_cluster = way + f"{self.launch_param['train_test_split']['cluster_type_model']}"
                self.current_way += f"{self.launch_param['train_test_split']['cluster_type_model']}"
            else:
                way_cluster = way
            
            self.current_way += self.split_hyper
            way_hyp = way_cluster + f'{self.split_hyper}'
            
            if os.path.exists(way):
                if os.path.exists(way_cluster):
                    if not os.path.exists(way_hyp):
                        os.mkdir(way_hyp)  
                else:
                    os.mkdir(way_cluster)
                    os.mkdir(way_hyp)
            else:
                os.mkdir(way)
                if self.type_split == 'cluster': 
                    os.mkdir(way_cluster)
                try:
                    os.mkdir(way_hyp)
                except FileExistsError:
                    pass
                
            with open(f'{way_hyp}/X_train.pickle', 'wb') as file:
                pickle.dump(self.X_train, file)
            with open(f'{way_hyp}/X_test.pickle', 'wb') as file:
                pickle.dump(self.X_test, file)
            with open(f'{way_hyp}/y_train.pickle', 'wb') as file:
                pickle.dump(self.y_train, file)
            with open(f'{way_hyp}/y_test.pickle', 'wb') as file:
                pickle.dump(self.y_test, file)
            with open(f'{way_hyp}/train_indexes.pickle', 'wb') as file:
                pickle.dump(self.train_ind, file)
            with open(f'{way_hyp}/test_indexes.pickle', 'wb') as file:
                pickle.dump(self.test_ind, file)
            with open(f'{way_hyp}/split.pickle', 'wb') as file:
                pickle.dump(self.split, file)

            # print(f'Запись разделений - {way_hyp}')
                
    def func_models(self):
        """
        Обучение модели, подбор гиперпараметров
        """
        self.type_model = self.launch_param['models']['type_model']

        way = f'{self.current_way}/{self.type_model}/threshold_{self.launch_param["models"]["threshold"]}'
        if os.path.exists(f'{way}/model.pickle'):
            with open(f'{way}/model.pickle', 'rb') as file:
                self.model = pickle.load(file)
            way_hyp = way
        else:
            self.model = ModelsClassical(type_class=self.launch_param['classification_type'],
                                         type_model=self.launch_param['models']['type_model'],
                                         y=self.launch_param['models']['y'])

            if self.launch_param['classification_type'] == 'multy':
                binary = False
            else:
                binary = True

            final_params = search_hyper(self.model, self.X_train, self.X_test, self.y_train, self.y_test,
                                        binary=binary, func_eval=self.launch_param['models']['func_eval'],
                                        aver=self.launch_param['models']['average'],
                                        **self.launch_param['models']['hyper_param'])
            final_params['random_state'] = self.launch_param['models']['random_state']

            self.model = ModelsClassical(type_class=self.launch_param['classification_type'],
                                         type_model=self.launch_param['models']['type_model'],
                                         y=self.launch_param['models']['y'],
                                         **final_params)
            self.model.fit(self.X_train, self.y_train, self.X_test, self.y_test)

            way = f'{self.current_way}/{self.type_model}'
            way_hyp = way + f'/threshold_{self.launch_param["models"]["threshold"]}'

            if os.path.exists(way):
                if not os.path.exists(way_hyp):
                    os.mkdir(way_hyp)
            else:
                os.mkdir(way)
                os.mkdir(way_hyp)

            # os.mkdir(way)
            with open(f'{way_hyp}/model.pickle', 'wb') as file:
                pickle.dump(self.model, file)

            # print(f'Запись модели - {way_hyp}')


        if self.launch_param['classification_type'] == 'binary' or \
        self.launch_param['models']['type_model'] == 'lgbm':
            self.log['model'] = {'type_of_model': type(self.model.model).__name__}
            self.log['model']['model_param'] = self.model.model.get_params()
        else:
            self.log['model'] = {'type_of_model': type(self.model.model.estimator).__name__}
            self.log['model']['model_param'] = self.model.model.estimator.get_params()

        self.current_way = way_hyp

    def func_eval(self):
        """
        Оценка модели
        """
        self.post_processing = PostProcessing(self.model, self.X_train, self.y_train,
                                              type_classification=self.launch_param['classification_type'],
                                              train_test_flag=True, X_2=self.X_test, y_2=self.y_test,
                                              average=self.launch_param['models']['average'],
                                              threshold=self.launch_param['models']['threshold'])

        self.log['evaluation'] = self.post_processing.dict_results

    def logging(self):
        """
        Функция для записи лога в json файл
        """
        with open(f'{self.current_way}/{self.log["experiment_number"]}_experiment_results.json', 'w') as file:
            json.dump(self.log, file, indent='\t')
