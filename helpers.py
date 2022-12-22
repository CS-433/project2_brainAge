import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV


def clean_data(i_dataset, df, del_col=[], mean=[], std=[]):
    X = df.copy()
    
    # Remove useless columns
    if (i_dataset == 1):
        X = X.drop(['Patient ID', 'Batch Process ID'], axis=1)
    elif (i_dataset == 2):
        X = X.drop(['Subject ID', 'Date of birth', 'Date of MRI scan'], axis=1)
    
    # Remove columns with only zeros
    if (len(del_col) == 0):
        del_col = (X != 0).any(axis=0)
    X = X.loc[:, del_col]
    
    # Replace zeros by nan
    X.replace(0, np.nan, inplace=True)
    
    # Replace Sex labels by values
    if (i_dataset == 1):
        X['Sex'].replace({'Male':1,'Female':2}, inplace=True)
        X = X.rename(columns={"Sex": "Gender"})
    
    # Categorical values
    cater = X[['Gender']]
    
    # Numerical values
    numer = X.drop(['Gender'], axis=1)
    
    # Replace nan values by mean of column
    if (len(mean) == 0 or len(std) == 0):
        mean = numer.mean()
        std = numer.std()
        std = std.fillna(1)
    numer = numer.fillna(mean)
                 
    
    #Scale each column in numer
    numer = (numer - mean)/std
    
    new_X = pd.concat([numer, cater], axis=1, join='inner')
    
    return new_X, del_col, mean, std

def standardize(x_train, x_test):
    # Standardize the training and testing sets according to training mean and std
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)

    x_train_scaled = (x_train - train_mean)/train_std
    x_test_scaled = (x_test - train_mean)/train_std

    return x_train_scaled, x_test_scaled


def predict(model, x_train, x_test):
    """ Compute model's predictions
    
    Args:
        model (model): Estimator
        x_train (np.array): Training labels of shape (N1, D).
        x_test (np.array): Testing labels of shape (N2, D).
        
    Returns:
        predict_train (np.array): Predictions for training data of shape (N1, ).
        predict_test (np.array): Predictions for testing data of shape (N2, ).

    """
    if len(x_train) > 0:
        predict_train = model.predict(x_train.drop(['Cluster'], axis=1, errors='ignore'))
    else:
        predict_train = []
    if len(x_test) > 0:
        predict_test = model.predict(x_test.drop(['Cluster'], axis=1, errors='ignore'))
    else:
        predict_test = []
    
    return predict_train, predict_test

def make_mi_scores(X, y):
    """ Compute mutual information scores
    
    Args:
        X (pd.DataFrame): Dataset of shape (N, D).
        y (np.array): Labels of shape (N, ).
        
    Returns:
        mi_scores (pd.Series): MI Scores of features

    """
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def compute_mae(y_train, y_test, predict_train, predict_test):
    """ Compute MAE scores
    
    Args:
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        predict_train (np.array): Predictions for training data of shape (N1, )
        predict_test (np.array): Predictions for testing data of shape (N2, )
        
    Returns:
        train_mae (float): Training MAE
        test_mae (float): Testing MAE

    """
    if len(predict_train) > 0:
        train_mae = mean_absolute_error(y_train, predict_train)
    else:
        train_mae = np.nan
    if len(predict_test) > 0:
        test_mae = mean_absolute_error(y_test, predict_test)
    else:
        test_mae = np.nan
    
    return train_mae, test_mae

def compute_r2(y_train, y_test, predict_train, predict_test):
    """ Compute R2 scores
    
    Args:
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        predict_train (np.array): Predictions for training data of shape (N1, )
        predict_test (np.array): Predictions for testing data of shape (N2, )
        
    Returns:
        train_r2 (float): Training R2 score
        test_r2 (float): Testing R2 score

    """
    if len(predict_train) > 0:
        train_r2 = r2_score(y_train, predict_train)
    else:
        train_r2 = np.nan
    if len(predict_test) > 0:
        test_r2 = r2_score(y_test, predict_test)
    else:
        test_r2 = np.nan
    
    return train_r2, test_r2

def plot_results(title, x_train, x_test, y_train, y_test, predict_train, predict_test):
    """ Plot results
    
    Args:
        title (str): Title of plots
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        predict_train (np.array): Predictions for training data of shape (N1, )
        predict_test (np.array): Predictions for testing data of shape (N2, )
        
    Returns:

    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel('True Age', fontsize = 15)
    ax1.set_ylabel('Predicted Age', fontsize = 15)
    ax1.set_title(title + ' - Train')
    ax2.set_xlabel('True Age', fontsize = 15)
    ax2.set_title(title + ' - Test')

    targets = [0, 1, 2]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        idx_train = np.where(x_train['Cluster'] == target)
        idx_test = np.where(x_test['Cluster'] == target)
        ax1.scatter(y_train.iloc[idx_train],
                    predict_train[idx_train],
                   c = color, 
                   s = 50, label=('Cluster ' + str(i)))
        ax2.scatter(y_test.iloc[idx_test],
                    predict_test[idx_test],
                   c = color, 
                   s = 50, label=('Cluster ' + str(i)))
    ax1.legend()
    ax1.grid()
    ax2.legend()
    ax2.grid()

    p1 = max(max(predict_train), max(np.array(y_train)))
    p2 = min(min(predict_train), min(np.array(y_train)))
    ax1.plot([p1, p2], [p1, p2], 'b-')
    ax2.plot([p1, p2], [p1, p2], 'b-')


def model_gs(model, param_grid): 
    """ Construct GridSearchCV object for model
    
    Args:
        model (model): Model to evaluate 
        param_grid (dict): List of parameters to iterate
        
    Returns:
        gs (GridSearchCV obj): GridSearchCV object for model
    """
    cv = RepeatedKFold(n_splits=3, n_repeats=5, random_state=42)
    
    # Construct pipeline
    pipe = Pipeline([
        ('clf', model)
    ])
    
    # Construct grid search
    gs = GridSearchCV(estimator=pipe,
        param_grid=param_grid,
        scoring='r2',
        cv=cv, verbose=10, n_jobs=-1, return_train_score = True)
    
    return gs

def train_model(title, gs, x_train, x_test, y_train, y_test, plot, criterion='r2'):
    """ Train model and compute its results
    
    Args:
        title (str): Title of plots, useless if plot=False
        gs (GridSeachCV obj): GridSearchCV object for the model 
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        plot (boolean): If True will plot results for each model
        criterion (str): Scoring criterion for the GridSearchCV 
        
    Returns:
        best_model (model): Best estimator
        best_score (float): Score of estimator during GridSearchCV
        train_r2 (float): Training R2 Score of estimator
        test_r2 (float): Testing R2 Score of estimator
        train_mae (float): Training MAE Score of estimator
        test_mae (float): Testing MAE Score of estimator
    """
    if (criterion == 'r2'):
        gs.scoring = 'r2'
    elif (criterion == 'mae'):
        gs.scoring = 'neg_mean_absolute_error'

    # Fit using grid search
    gs.fit(x_train.drop(['Cluster'], axis=1, errors='ignore'), y_train['Age'])

    # Compute predictions
    predict_train, predict_test = predict(
        gs.best_estimator_,
        x_train,
        x_test
    )
      
    # Compute MAE and R2 scores
    train_mae, test_mae = compute_mae(y_train['Age'], y_test['Age'], predict_train, predict_test)
    train_r2, test_r2 = compute_r2(y_train['Age'], y_test['Age'], predict_train, predict_test)
    
    # If True, plot the results
    if (plot == True):
        plot_results(title, x_train, x_test, y_train, y_test, predict_train, predict_test)

    # Select best estimator
    best_model = gs.best_estimator_.steps[0][1]
    # Select best score during GridSearchCV for the estimator
    best_score = gs.best_score_

    return best_model, best_score, train_r2, test_r2, train_mae, test_mae

def train_all(x_train, x_test, y_train, y_test, models, plot=True, title='global'):
    """ Train each model in list of models and compute the results of each model
    
    Args:
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        models (dict): Dictionnary listing models to fit
        plot (boolean): If True will plot results for each model
        title (str): Title of plots, useless if plot=False
        
    Returns:
        results (pd.DataFrame): Results of each model on training and testing data
    """
    results = pd.DataFrame(columns=["Best score", "Train R2",  "Test R2", "Train MAE", "Test MAE"])
    
    #Test for each model
    for model_name in models:
        results.loc[model_name] = train_model(
            model_name + " (" + title + ")",
            models[model_name],
            x_train,
            x_test,
            y_train,
            y_test,
            plot
        )[1:]
        
    return results

def train_cluster(i, x_train, x_test, y_train, y_test, models, plot=True):
    """ Train each model in list of models and compute the results of each model
    
    Args:
        i (integer): Cluster number
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        models (dict): Dictionnary listing models to fit
        plot (boolean): If True will plot results for each model
        
    Returns:
        results (pd.DataFrame): Results of each model on training and testing data
    """

    # Select data corresponding to Cluster i
    idx_train = np.where(x_train["Cluster"]==i)
    x_train_cluster = x_train.iloc[idx_train].reset_index(drop=True)
    y_train_cluster = y_train.iloc[idx_train].reset_index(drop=True)

    idx_test = np.where(x_test["Cluster"]==i)
    x_test_cluster = x_test.iloc[idx_test].reset_index(drop=True)
    y_test_cluster = y_test.iloc[idx_test].reset_index(drop=True)
    
    # Train models and compute the results
    results = train_all(x_train_cluster, x_test_cluster, y_train_cluster, y_test_cluster, models, plot, 'Cluster ' + str(i))
    
    return results

def find_best_model(x_train, x_test, y_train, y_test, models):
    """ Train each model in list of models and find the best ones
        One with best Test R2 and one with best Test MAE
    
    Args:
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        models (dict): Dictionnary listing models to fit
        
    Returns:
        best_model (dict): Dictionnary listing best model per criterion
        best_results (dict): Dictionnary listing results of best model per criterion
    """
    best_results = {
        'mae': [0, -100, 0, 100],
        'r2': [0, -100, 0, 100],
    }
    best_model = {}

    #Test for each model
    for model_name in models:
        # Train model and compute results
        results_model = train_model(
            '',
            models[model_name],
            x_train,
            x_test,
            y_train,
            y_test,
            plot=False
        )
        # If Test R2 is better, make it best model for R2 scoring
        if (best_results['r2'][1] < results_model[3]):
            best_results['r2'] = results_model[2:]
            best_model['r2'] = results_model[0]
        
        # If Test MAE is better, make it best model for MAE scoring
        if (best_results['mae'][3] > results_model[5]):
            best_results['mae'] = results_model[2:]
            best_model['mae'] = results_model[0]

    return best_model, best_results

def find_best_models(models, x_train, x_test, y_train, y_test):
    """ Train global prediction model and local prediction models
        then find best model for each cluster (global or local)
    
    Args:
        models (dict): Dictionnary listing models to fit
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        
    Returns:
        best_models (dict): Dictionnary of array, per criterion, listing best models per cluster
        global_score (pd.DataFrame): Results of the global prediction model
    """
    
    n_clusters = x_train['Cluster'].unique().size
    criterions = ['r2', 'mae']
    
    # Split data in clusters
    x_train_clusters = []
    x_test_clusters = []
    y_train_clusters = []
    y_test_clusters = []
    
    for i in range(n_clusters):
        idx_train = np.where(x_train["Cluster"]==i)
        x_train_clusters.append(x_train.iloc[idx_train].reset_index(drop=True))
        y_train_clusters.append(y_train.iloc[idx_train].reset_index(drop=True))

        idx_test = np.where(x_test["Cluster"]==i)
        x_test_clusters.append(x_test.iloc[idx_test].reset_index(drop=True))
        y_test_clusters.append(y_test.iloc[idx_test].reset_index(drop=True))
        
        
    # Train global model and compute its results
    model_global, results_global_model = find_best_model(x_train, x_test, y_train, y_test, models)
    
    # Format global results
    global_score = {}
    for crit in criterions:
        global_score[crit] = np.concatenate([np.array([model_global[crit].__class__.__name__]), np.array(results_global_model[crit], dtype='U5')])
    
    global_score = pd.DataFrame(
        global_score.values(),
        index=[f'Best {key.upper()}' for key in global_score.keys()],
        columns=["Model", "Train R2",  "Test R2", "Train MAE", "Test MAE"]
    )

    print('\033[1m', '-- Global Model --', '\033[0m')
    for key in model_global.keys():
        print(f'Best {key.upper()} : {model_global[key]}')
    print('\033[1m', global_score,'\033[0m')
    
    
    # Train local model for each cluster
    model_clusters = {}
    results_clusters = {}
    results_global = {}
    for i in range(n_clusters):
        # Find best local model and compute its results
        model_i, results_i = find_best_model(
            x_train_clusters[i],
            x_test_clusters[i],
            y_train_clusters[i],
            y_test_clusters[i],
            models)
        
        predict_train_i = {}
        predict_test_i = {}
        train_mae_i = {}
        test_mae_i = {}
        train_r2_i = {}
        test_r2_i = {}
        # For each cluster
        for crit in criterions:
            if crit not in model_clusters:
                model_clusters[crit] = []
            # Save local model
            model_clusters[crit].append(model_i[crit])
            if crit not in results_clusters:
                results_clusters[crit] = []
            # Save results of local model
            results_clusters[crit].append(results_i[crit])   
            
            # Compute cluster predictions based on global model
            predict_train_i[crit], predict_test_i[crit] = predict(
                model_global[crit],
                x_train_clusters[i],
                x_test_clusters[i]
            )

            # Compute MAE and R2 scores of the global model predictions for the cluster
            train_mae_i[crit], test_mae_i[crit] = compute_mae(
                y_train_clusters[i],
                y_test_clusters[i],
                predict_train_i[crit],
                predict_test_i[crit])
            train_r2_i[crit], test_r2_i[crit] = compute_r2(
                y_train_clusters[i],
                y_test_clusters[i],
                predict_train_i[crit],
                predict_test_i[crit])
            
            if crit not in results_global:
                results_global[crit] = []
            # Save results of global model
            results_global[crit].append([train_r2_i[crit], test_r2_i[crit], train_mae_i[crit], test_mae_i[crit]])
    
    print('-- Global Model on clusters')
    for crit in criterions:
        print(pd.DataFrame(
            results_global[crit],
            index=[f'Best {crit.upper()} - Cluster {i}' for i in range(n_clusters)],
            columns=["Train R2",  "Test R2", "Train MAE", "Test MAE"]
        ))
    print('-- Clusters Models --')
    for crit in criterions:
        for i in range(n_clusters):
            print(f'Best {crit} - Cluster {i} : {model_clusters[crit][i]}')

        print(pd.DataFrame(
            results_clusters[crit],
            index=[f'Best {crit.upper()} - Cluster {i}' for i in range(n_clusters)],
            columns=["Train R2",  "Test R2", "Train MAE", "Test MAE"]
        ))
    
    # Select best model for each cluster
    best_models = {}
    best_results = {}
    best_origin = {}
    for crit in criterions:
        best_models[crit] = []
        best_results[crit] = []
        best_origin[crit] = {}
    for i in range(n_clusters):
        # If Test R2 is better for global model than local model
        if (results_global['r2'][i][1] > results_clusters['r2'][i][1]):
            # Assign global model for this cluster, for R2 scoring criterion
            best_models['r2'].append(model_global['r2'])
            best_results['r2'].append(results_global['r2'][i])
            best_origin['r2'][i] = 'global'
        else:
            # Assign local model for this cluster, for R2 scoring criterion
            best_models['r2'].append(model_clusters['r2'][i])
            best_results['r2'].append(results_clusters['r2'][i])
            best_origin['r2'][i] = 'local'

        # If Test MAE is better for global model than local model
        if (results_global['mae'][i][3] < results_clusters['mae'][i][3]):
            # Assign global model for this cluster, for MAE scoring criterion
            best_models['mae'].append(model_global['mae'])
            best_results['mae'].append(results_global['mae'][i])
            best_origin['mae'][i] = 'global'
        else:
            # Assign local model for this cluster, for MAE scoring criterion
            best_models['mae'].append(model_clusters['mae'][i])
            best_results['mae'].append(results_clusters['mae'][i])
            best_origin['mae'][i] = 'local'

    print('-- Best Models --')
    for crit in criterions:
        for i in range(n_clusters):
            print(f'Best {crit} - Cluster {i} : {best_models[crit][i]}')
        
        print(pd.DataFrame(
            best_results[crit],
            index=[f'Best {crit.upper()} - Cluster {i}' for i in range(n_clusters)],
            columns=["Train R2",  "Test R2", "Train MAE", "Test MAE"]
        ))

    print('-- Optimization results --')
    for crit in criterions:
        for i in range(n_clusters):
            print(f'Best {crit} - Cluster {i} : {best_origin[crit][i]}')
    
    return best_models, global_score

def find_best_results(best_models, x_train, x_test, y_train, y_test):
    """ Compute results for the optimized prediction model
    
    Args:
        best_models (dict): Dictionnary of array, per criterion, listing best models per cluster
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        
    Returns:
        results (pd.DataFrame): Results of the optimized prediction model
    """
    n_clusters = x_train['Cluster'].unique().size
    criterions = ['r2', 'mae']
    
    # Split data in clusters
    x_train_clusters = []
    x_test_clusters = []
    y_train_clusters = []
    y_test_clusters = []
    predict_train_clusters = {}
    predict_test_clusters = {}
    for crit in criterions:
        predict_train_clusters[crit] = []
        predict_test_clusters[crit] = []
    
    # For each cluster
    for i in range(n_clusters):
        idx_train = np.where(x_train["Cluster"]==i)
        x_train_clusters.append(x_train.iloc[idx_train].reset_index(drop=True))
        y_train_clusters.append(y_train.iloc[idx_train].reset_index(drop=True))

        idx_test = np.where(x_test["Cluster"]==i)
        x_test_clusters.append(x_test.iloc[idx_test].reset_index(drop=True))
        y_test_clusters.append(y_test.iloc[idx_test].reset_index(drop=True))
        
        predict_train_i = {}
        predict_test_i = {}
        # For each criterion
        for crit in criterions:
            # Compute cluster predictions
            predict_train_i[crit], predict_test_i[crit] = predict(
                best_models[crit][i],
                x_train_clusters[i],
                x_test_clusters[i]
            )
            predict_train_clusters[crit].append(predict_train_i[crit])
            predict_test_clusters[crit].append(predict_test_i[crit])
    
    # Order labels by cluster (align them with predictions)
    y_train_merge = np.concatenate(y_train_clusters)
    y_test_merge = np.concatenate(y_test_clusters)
    
    predict_train_merge = {}
    predict_test_merge = {}
    train_mae = {}
    test_mae = {}
    train_r2 = {}
    test_r2 = {}
    
    # For each criterion
    for crit in criterions:
        predict_train_merge[crit] = np.concatenate(predict_train_clusters[crit])
        predict_test_merge[crit] = np.concatenate(predict_test_clusters[crit])

        # Compute MAE and R2 scores
        train_mae[crit], test_mae[crit] = compute_mae(
            y_train_merge,
            y_test_merge,
            predict_train_merge[crit],
            predict_test_merge[crit]
        )
        train_r2[crit], test_r2[crit] = compute_r2(
            y_train_merge,
            y_test_merge,
            predict_train_merge[crit],
            predict_test_merge[crit]
        )

    # Format results
    results = pd.DataFrame(columns=["Train R2",  "Test R2", "Train MAE", "Test MAE"])
    for crit in criterions:
        results = results.append(
            pd.DataFrame(
                np.array([train_r2[crit], test_r2[crit], train_mae[crit], test_mae[crit]]).reshape(1,-1),
                index=[f'Best {crit}'],
                columns=results.columns
            ))
    
    print('\033[1m', results,'\033[0m')
    
    return results

def optimize_clusters(models, x_train, x_test, y_train, y_test):
    """ Apply optimization process to clusters prediction models
    
    Args:
        models (dict): Dictionnary listing models to fit
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, ).
        y_test (np.array): Testing labels of shape (N2, ).
        
    Returns:
        optimized_results (pd.DataFrame): Results of the optimized prediction model
        results_global (pd.DataFrame): Results of the global prediction model
        best_models (np.array): Best prediction model per cluster
    """
    # Train global model and local models and compute the global results. Find the best model per cluster
    best_models, results_global = find_best_models(
        models,
        x_train,
        x_test,
        y_train,
        y_test
    )
    # Compute results of the optimized prediction model
    optimized_results = find_best_results(
        best_models,
        x_train,
        x_test,
        y_train,
        y_test
    )
    return optimized_results, results_global, best_models

def KM_clustering(x_train, x_test, n_clusters=3):
    """ Apply K-Means clustering to datasets
    
    Args:
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        n_clusters (integer): Number of clusters to compute
    Returns:
        clustered_train (np.array): x_train with Cluster feature (N1, D+1)
        clustered_test (np.array): x_test with Cluster feature (N2, D+1)
    """
    clustered_train = x_train.copy()
    clustered_test = x_test.copy()
    kmeanModel = KMeans(n_clusters=n_clusters).fit(clustered_train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    clustered_train['Cluster'] = kmeanModel.labels_
    clustered_test['Cluster'] = kmeanModel.predict(clustered_test.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    return clustered_train, clustered_test

def GMM_clustering(x_train, x_test, n_clusters=3):
    """ Apply Gaussian Mixture Model clustering to datasets
    
    Args:
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        n_clusters (integer): Number of clusters to compute
    Returns:
        clustered_train (np.array): x_train with Cluster feature (N1, D+1)
        clustered_test (np.array): x_test with Cluster feature (N2, D+1)
    """
    clustered_train = x_train.copy()
    clustered_test = x_test.copy()
    gmm = GaussianMixture(n_components=n_clusters).fit(clustered_train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    clustered_train['Cluster'] = gmm.predict(clustered_train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))
    clustered_test['Cluster'] = gmm.predict(clustered_test.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    return clustered_train, clustered_test

def PLS_regression(x_train, x_test, y_train, y_test, n_components):
    """ Apply Partial Least Square regression to datasets to reduce dimensionality
    
    Args:
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, )
        y_test (np.array): Testing labels of shape (N2, )
        n_components (integer): Number of reduced features to compute
    Returns:
        pls_train (np.array): x_train with features of reduced shape (N1, d)
        pls_test (np.array): x_test with features of reduced shape (N2, d)
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(x_train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'), y_train)

    new_x_train = pls.transform(x_train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))
    new_x_test = pls.transform(x_test.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))
    pls_train = pd.DataFrame(data = new_x_train, columns = ["class%02d" %i for i in range(1,n_components+1)])
    pls_test = pd.DataFrame(data = new_x_test, columns = ["class%02d" %i for i in range(1,n_components+1)])

    return pls_train, pls_test

def PCA_decomposition(x_train, x_test, y_train, y_test, n_components):
    """ Apply Principal Components Analysis to datasets to reduce dimensionality
    
    Args:
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N1, )
        y_test (np.array): Testing labels of shape (N2, )
        n_components (integer): Number of principal components to compute
    Returns:
        pc_train (np.array): x_train with features of reduced shape (N1, d)
        pc_test (np.array): x_test with features of reduced shape (N2, d)
    """
    pca = PCA(n_components=n_components)

    # Fit PCA to training data and compute the n_components for training data
    pc_train = pca.fit_transform(x_train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))
    # Compute the n_components for testing data
    pc_test = pca.transform(x_test.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    # Label the principal components
    pc_train = pd.DataFrame(data = pc_train, columns = ["PC%02d" %i for i in range(1,pca.n_components_+1)])
    pc_test = pd.DataFrame(data = pc_test, columns = ["PC%02d" %i for i in range(1,pca.n_components_+1)])

    return pc_train, pc_test

def filter_mi_scores(x_train, x_test, y_train, threshold=0):
    """ Filter features with mutual information scoring 
    
    Args:
        x_train (np.array): Training dataset of shape (N1, D).
        x_test (np.array): Testing dataset of shape (N2, D).
        y_train (np.array): Training labels of shape (N, )
        threshold (float): Filtering threshold for the scores (min)
    Returns:
        x_train_filtered (np.array): x_train with filtered features of shape (N1, d)
        x_test_filtered (np.array): x_test with filtered features of shape (N2, d)
    """

    # Compute MI Scores of training data
    mi_score_train = make_mi_scores(x_train.drop(['Cluster'], axis=1, errors='ignore'), y_train['Age'])

    # Select features according to threshold
    keep_features = (mi_score_train[mi_score_train > threshold].index.to_list())
    
    # If Cluster in training set, keep Cluster feature
    if 'Cluster' in x_train.columns:
        keep_features.append('Cluster')

    # Filter the datasets
    x_train_filtered = x_train.copy()[keep_features]
    x_test_filtered = x_test.copy()[keep_features]

    return x_train_filtered, x_test_filtered

def filter_lasso(x_train, x_test, y_train, threshold=0):
    """ Filter features with Lasso scoring 
    
    Args:
        x_train (np.array): Training dataset of shape (N, D).
        x_test (np.array): Testing dataset of shape (N, D).
        y_train (np.array): Training labels of shape (N, )
        threshold (float): Filtering threshold for the scores (min)
    Returns:
        x_train_filtered (np.array): x_train with filtered features of shape (N, d)
        x_test_filtered (np.array): x_test with filtered features of shape (N, d)
    """

    # Fit Lasso regressor to training data
    reg = LassoCV(cv=5, random_state=0, n_jobs=-1)
    reg.fit(x_train, y_train)
    # List absolute values of coeficients of features
    coef = pd.Series(np.absolute(reg.coef_), index = x_train.columns)
    # Select the features according to the threshold
    lasso_features = coef.iloc[coef.to_numpy() > threshold].index

    print("Lasso picked " + str(len(lasso_features)) + " variables and eliminated the other " +  str(len(coef)-len(lasso_features))  + " variables")

    # Filter the datasets
    x_train_filtered = x_train[lasso_features]
    x_test_filtered = x_test[lasso_features]

    return x_train_filtered, x_test_filtered