from collections import Counter
import re

from pandarallel import pandarallel
from ufal.udpipe import Model, Pipeline
from sklearn.base import TransformerMixin
import pymorphy2 as pm
import nltk
import numpy as np
import jamspell
from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,
    DatesExtractor,

    Doc
)


def choose_target(data, target=None):
    """
    Функция для фильтрации датасета с указанными классами. По-умолчанию возвращает тот же датасет.
        data: pandas.DataFrame
            Исходный датасет
        target: tuple/list, default=None
            Кортеж или список классов, объекты которых необходимо оставить в датасете
    
        return: pandas.DataFrame
            Отфильтрованный датасет
    """
    if target:
        data = data[np.isin(data['class'], target)]
        
    return data


def count_classes(y, dict_classes, text=''):
    """
    Функция для вывода на экран содержания каждого класса в выборке в процентах
        y: pandas.Series/numpy.ndarray
            Вектор целевых переменных
        dict_classes: dict
            Словарь с расшифровкой целевых переменных
        text: str, default=''
            Заголовок
    """
    all_count =  y.size
    print(f'{text}')
    for target in sorted(y.unique()):
        count_target = y[y == target].shape[0]
        print(f'Соотношение {dict_classes[target]} ({target}) - {round(count_target / all_count * 100, 2)} %')
        
        
def count_words(text_list):
    """
    Функция подсчета слов в корпусе
        text_list: list
            Cписок текстов в корпусе
        
        return: Counter
            Cловарь-счетчик
    """
    count_dict = {}
    for text in text_list:
        list_words = text.split(' ')
        if count_dict:
            count_dict += Counter(list_words)
        else:
            count_dict = Counter(list_words)
    return count_dict        


class TextPrepare(TransformerMixin):
    """
    Класс для предобработки текста. Включает в себя очистку текста от спецсимволов, очистку от стоп-слов,
    стемминг, лемматизацию, обработку парсером UDPipe. Класс содержит в себе методы fit и transform для
    использования объектов класса в пайплайнах sklearn.
    """
    def __init__(self, stopwords, lang='ru', type_preproc=None, speech_2_vec_lemm=False, way_to_udp=None, 
                 keep_pos_udp=True, keep_punct_udp=False,
                 model_checker='ru_small.bin', parallel_workers=20):
        """
        Очистка текста от спецсимволов и стоп-слов
            stopwords: list
                Список стоп-слов
            lang: {'ru', 'eng'}, default='ru'
                Язык текста
            type_preproc: {'stem', 'lemm', 'udp', None}, default=None
                Тип предобработки - стемминг, лемматизация, обработка UDPipe парсером или ничего.
                Стемминг осуществляется через библиотеку nltk (SnowballStemmer).
                Лемматизация осуществляется через библиотеку pymorphy2. К лемме также можно добавить часть речи из тэгов pymorphy2 через нижнее подчеркивание '_' - 'делать_VERB'.
                Обработка UDPipe парсером осуществляется через библиотеку ufal.udpipe. Обработка включает в себя лемматизацию с возможностью добавления части речи через нижнее подчеркивание (тэги частей речи соответствуют  формату Universal PoS Tags) '_' - 'делать_VERB', также существует возможность сохранения пунктуации.
            speech_2_vec_lemm: bool, default=False
                Флаг для добавления части речи в предобработке с лемматизацией (type_preproc='lemm')
            way_to_udp: str, default=None
                Путь до обученной модели UDPipe
            keep_pos_udp: bool, default=True
                Флаг добавления части речи к лемме
            keep_punct_udp: bool, default=False
                Флаг сохранения пунктуации
            model_checker: str, default='ru_small.bin'
                Путь до модели проверки орфографии jamspell
            parallel_workers: int, default=20
                Кол-во процессорорв для распараллеливания
        """
        self.lang = lang

        self.replace_by_space_re = re.compile('[/(){}\[\]\|@\.,:;!?-]|[\s]')  # шаблон под замену данных символов на пробел
        self.bad_symbols_re = re.compile('[^0-9а-я #+_]')               # шаблон под удаление всех символов кроме этих
        self.stopwords_re = r'\b' + r'\b|\b'.join(stopwords) + r'\b'    # шаблон для чистки стоп-слов
        
        self.type_preproc = type_preproc
        self.speech_2_vec_lemm = speech_2_vec_lemm
        self.way_to_udp = way_to_udp
        self.keep_pos_udp = keep_pos_udp
        self.keep_punct_udp = keep_punct_udp
        self.corrector = jamspell.TSpellCorrector()
        self.corrector.LoadLangModel(model_checker)
        self.parallel_workers = parallel_workers
        
    
    def fit(self, X, y=None):
        """
        Пустой метод, нужен для использования объектов класса в пайплайнах sklearn
        """
        self.X_init = X
        
        return self
    
    def transform(self, X, y=None):
        """
        Метод для преобразования текста. Распараллеливается с помощью библиотеки pandarallel. 
        Применяется ко всему объекту Series с помощью метода apply.
            X: pandas.Series
                Корпус текста для преобразования 
                
            return: Series
                Обработанный корпус текста
        """
        X = X.apply(self.corrector.FixFragment)

        if self.parallel_workers == 1:
            return X.apply(self.__test_prepare)
        else:
            pandarallel.initialize(nb_workers=self.parallel_workers)
            return X.parallel_apply(self.__test_prepare)

    def __test_prepare(self, text):
        """
        Функция предобработки текста для каждого отдельного текста. 
        Имеет те же аргументы, что и именнованные аргументы метода __init__.
            text: str
                Принимает на вход документ текста
                
            return: str
                Возвращает обработанную строку текста
        """
        # text = self.__replace_name_person_date(text)

        try:
            text = text.lower()
        except AttributeError:
            print('Ошибка на этом тексте:')
            print(text)

        text = text.lower()
        text = re.sub(self.replace_by_space_re, ' ', text)
        text = re.sub(self.bad_symbols_re, '', text)
        text = re.sub(self.stopwords_re, '', text)
        text_list = list(filter(lambda x: x != '', text.split(' ')))

        if self.type_preproc == 'stem':
            new_text_list = self.__stemmer(text_list)
        elif self.type_preproc == 'lemm':
            new_text_list = self.__lemmattization(text_list)
        elif self.type_preproc == 'udp':
            new_text_list = self.__udp(text)
        else:
            new_text_list = text_list

        return ' '.join(new_text_list)

    # def __replace_name_person_date(self, text):
    #     segmenter = Segmenter()
    #     morph_vocab = MorphVocab()
    #
    #     emb = NewsEmbedding()
    #     morph_tagger = NewsMorphTagger(emb)
    #     syntax_parser = NewsSyntaxParser(emb)
    #     ner_tagger = NewsNERTagger(emb)
    #
    #     names_extractor = NamesExtractor(morph_vocab)
    #     dates_extractor = DatesExtractor(morph_vocab)
    #
    #     doc = Doc(text)
    #     doc.segment(segmenter)
    #     doc.tag_morph(morph_tagger)
    #     doc.parse_syntax(syntax_parser)
    #     doc.spans
    #     doc.tag_ner(ner_tagger)
    #
    #     for span in doc.spans:
    #         span.normalize(morph_vocab)
    #
    #     for span in doc.spans:
    #         if span.type == PER:
    #             span.extract_fact(names_extractor)
    #
    #     name_dict = {_.text: _.fact.as_dict for _ in doc.spans if _.fact}
    #
    #     for name in name_dict.keys():
    #         text = re.sub(name, 'заменаимени', text)
    #
    #     for match in dates_extractor(text):
    #         text = text[:match.start] + ' ' + 'заменадаты' + ' ' + text[(match.stop + 1):]
    #
    #     return text

    def __stemmer(self, text_list):
        """
        Метод для стемминга
            text_list: list
                Список токенов
                
            return: list
                Список токенов после стемминга
        """
        if self.lang == 'ru':
            stem = nltk.stem.SnowballStemmer('russian')
        elif self.lang == 'ru':
            stem = nltk.stem.SnowballStemmer('english')
        return [stem.stem(word) for word in text_list]
    
    def __lemmattization(self, text_list):
        """
        Метод для лемматизации
            text_list: list
                Список токенов
            
            return: list
                Список токенов после лемматизации
        """
        lemm = pm.MorphAnalyzer()
        final_list = []
        for word in text_list:
            parse = lemm.parse(word)[0]
            if self.speech_2_vec_lemm:
                part_of_speech = str(parse.tag).split(',')[0].split(' ')[0]
                final_list.append(f'{parse.normal_form}_{part_of_speech}')
            else:
                final_list.append(parse.normal_form)
            
        return final_list
        
    def __udp(self, text):
        """
        Метод для обработки текста с помощью синтаксического парсера UDPipe
            text: str
                Текст для обработки
            
            return: list
                Список токенов после обработки
        """
        model = Model.load(self.way_to_udp)
        process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

        entities = {'PROPN'}
        named = False
        memory = []
        mem_case = None
        mem_number = None
        tagged_propn = []

        # обрабатываем текст, получаем результат в формате conllu:
        processed = process_pipeline.process(text)

        # пропускаем строки со служебной информацией:
        content = [l for l in processed.split('\n') if not l.startswith('#')]

        # извлекаем из обработанного текста леммы, тэги и морфологические характеристики
        tagged = [w.split('\t') for w in content if w]

        for t in tagged:
            if len(t) != 10:
                continue
            (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
            if not lemma or not token:
                continue
            if pos in entities:
                if '|' not in feats:
                    tagged_propn.append('%s_%s' % (lemma, pos))
                    continue
                morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}
                if 'Case' not in morph or 'Number' not in morph:
                    tagged_propn.append('%s_%s' % (lemma, pos))
                    continue
                if not named:
                    named = True
                    mem_case = morph['Case']
                    mem_number = morph['Number']
                if morph['Case'] == mem_case and morph['Number'] == mem_number:
                    memory.append(lemma)
                    if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
                        named = False
                        past_lemma = '::'.join(memory)
                        memory = []
                        tagged_propn.append(past_lemma + '_PROPN ')
                else:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN ')
                    tagged_propn.append('%s_%s' % (lemma, pos))
            else:
                if not named:
                    if pos == 'NUM' and token.isdigit():  # Заменяем числа на xxxxx той же длины
                        lemma = 'x' * len(token)
                    tagged_propn.append('%s_%s' % (lemma, pos))
                else:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN ')
                    tagged_propn.append('%s_%s' % (lemma, pos))
#                     tagged_propn.append('NAME_PROPN')

        if not self.keep_punct_udp:
            tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PUNCT']
        if not self.keep_pos_udp:
            tagged_propn = [word.split('_')[0] for word in tagged_propn]
            
        return tagged_propn
   