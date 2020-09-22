from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score
import sklearn


def search_hyper(model, X_train, X_test, y_train, y_test,
                 binary=True, func_eval=f1_score, aver='binary', **kwargs):
    """
    Функция для подбора гиперпараметров в модели. Для регрессии и метода опорных векторов работает поиск по сетке.
    Для градиетного бустинга работает Баесовский оптимизатор (необходио передать 2 аргумента - params и bounds)
        model: object of model
            Модель МО
        X_train: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
            Тренировочный датасет
        X_test: pandas.DataFrame/numpy.ndarray/scipy.sparse.csr.csr_matrix
            Тестовый датасет
        y_train: pandas.Series/numpy.ndarray
            ЦП соответствующая тренироврочному датасету
        y_test: pandas.Series/numpy.ndarray
            ЦП соответствующая тестовому датасету
        binary: bool, default=True
            Флаг бинарной классификации (в случае False - многоклассовая классификация)
        func_eval: object of eval, default=sklearn.metrics.f1_score
            Функция для оценки метрики
        aver: {'micro', 'macro', 'weighted', 'binary'}, default='binary'
            Параметр, необходимый для усредения метрики для многоклассовой классификации
        **kwargs:
            Словари с гиперпараметрами
            
        return: dict
            Словарь с подобранными гиперпараметрами
    """
    if model.type_model == 'lin':
        C = kwargs['C']
        penalty = ['l1', 'l2']
        eval_dict = {}
           
        for pen in penalty:
            for c in C:
                if pen == 'l1':
                    solv = 'liblinear'
                else:
                    solv = 'lbfgs'
                    
                params = {'C': c,
                          'penalty': pen, 
                          'solver': solv}    
                 
                if type(model.model) == sklearn.multiclass.OneVsRestClassifier:
                    model.model.estimator.set_params(**params)
                else:
                    model.model.set_params(**params)
                    
                model.fit(X_train, y_train)
                    
                y_pred = model.predict(X_test)              
               
                eval_dict[func_eval(y_test, 
                                    y_pred, 
                                    average=aver)] = {'penalty': pen, 
                                                      'solver': solv,
                                                      'C': c}
                
                max_value = max(eval_dict)
                
        return eval_dict[max_value]
    elif model.type_model == 'lgbm':
        def __evaluate(**new_params):
            """
            Фукнция для баесовского оптимизатора, которая делит на фолды, обучает, предсказывает и выдает оценку
            new_params - словарь с гиперпараметрами

            return - полнота
            """
            try:
                new_params['n_estimators'] = int(new_params['n_estimators'])
            except KeyError:
                pass
            try:
                new_params['num_leaves'] = int(new_params['num_leaves'])
            except KeyError:
                pass
            try:
                new_params['min_data_in_leaf'] = int(new_params['min_data_in_leaf'])
            except KeyError:
                pass
            try:
                new_params['max_depth'] = int(new_params['max_depth'])
            except KeyError:
                pass
            try:
                new_params['max_bin'] = int(new_params['max_bin'])
            except KeyError:
                pass
            
            params.update(new_params)
#             print(new_params)
            model.model.set_params(**new_params)

            model.fit(X_train, y_train, X_test, y_test)
            y_pred = model.predict(X_test)

            return func_eval(y_test, y_pred, average=aver)
        
        if binary:
            objective = 'binary'
            metric = 'binary_logloss'
        else:
            objective = 'multiclass'
            metric = 'multi_logloss'
        
        params = kwargs['params']
        bounds = kwargs['bounds']
        verbose = kwargs['verbose']
    
        bo = BayesianOptimization(__evaluate, pbounds=bounds, verbose=verbose)
        bo.maximize(init_points=5, n_iter=10)
        
        par = bo.max['params']

        par['num_leaves'] = int(par['num_leaves'])
        par['min_data_in_leaf'] = int(par['min_data_in_leaf'])
        par['max_depth'] = int(par['max_depth'])
        par['n_estimators'] = int(par['n_estimators'])
        par['num_leaves'] = int(par['num_leaves'])
        par['min_sum_hessian_in_leaf'] = int(par['min_sum_hessian_in_leaf'])
        par['objective'] = objective
        par['metric'] = metric
        par['max_bin'] = params['max_bin']
        
        return par
