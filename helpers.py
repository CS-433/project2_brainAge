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
        std = std.fillna( 1)
    numer = numer.fillna(mean)
                 
    
    #Scale each column in numer
    numer = (numer - mean)/std
    
    new_X = pd.concat([numer, cater], axis=1, join='inner')
    
    return new_X, del_col, mean, std


def predict(model, x_train, x_test):
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
                   s = 50)
        ax2.scatter(y_test.iloc[idx_test],
                    predict_test[idx_test],
                   c = color, 
                   s = 50)
    ax1.legend(['Cluster 0', 'Cluster 1', 'Cluster 2'])
    ax1.grid()
    ax2.legend(['Cluster 0', 'Cluster 1', 'Cluster 2'])
    ax2.grid()

    p1 = max(max(predict_train), max(np.array(y_train)))
    p2 = min(min(predict_train), min(np.array(y_train)))
    ax1.plot([p1, p2], [p1, p2], 'b-')
    ax2.plot([p1, p2], [p1, p2], 'b-')


def model_gs(model, param_grid): 
    
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
    
    if (plot == True):
        plot_results(title, x_train, x_test, y_train, y_test, predict_train, predict_test)

    return gs.best_estimator_.steps[0][1], gs.best_score_, train_r2, test_r2, train_mae, test_mae

def train_all(x_train, x_test, y_train, y_test, models, plot=True, title='global'):
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
    idx_train = np.where(x_train["Cluster"]==i)
    x_train_cluster = x_train.iloc[idx_train].reset_index(drop=True)
    y_train_cluster = y_train.iloc[idx_train].reset_index(drop=True)

    idx_test = np.where(x_test["Cluster"]==i)
    x_test_cluster = x_test.iloc[idx_test].reset_index(drop=True)
    y_test_cluster = y_test.iloc[idx_test].reset_index(drop=True)
    
    results = train_all(x_train_cluster, x_test_cluster, y_train_cluster, y_test_cluster, models, plot, 'Cluster ' + str(i))
    
    return results

def find_best_model(x_train, x_test, y_train, y_test, models):

    best_results = {
        'mae': [0, -100, 0, 100],
        'r2': [0, -100, 0, 100],
    }
    best_model = {}
    #Test for each model
    for model_name in models:
        results_model = train_model(
            '',
            models[model_name],
            x_train,
            x_test,
            y_train,
            y_test,
            plot=False
        )
        
        if (best_results['r2'][1] < results_model[3]):
            best_results['r2'] = results_model[2:]
            best_model['r2'] = results_model[0]
            
        if (best_results['mae'][3] > results_model[5]):
            best_results['mae'] = results_model[2:]
            best_model['mae'] = results_model[0]

    return best_model, best_results

def find_best_models(models, x_train, x_test, y_train, y_test):
    
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
        
        
    # Train global model
    model_global, global_score = find_best_model(x_train, x_test, y_train, y_test, models)
    
    for crit in criterions:
        global_score[crit] = np.concatenate([np.array([model_global[crit].__class__.__name__]), np.array(global_score[crit], dtype='U4')])
    global_score = pd.DataFrame(
        global_score.values(),
        index=[f'Best {key.upper()}' for key in global_score.keys()],
        columns=["Model", "Train R2",  "Test R2", "Train MAE", "Test MAE"]
    )
    print('\033[1m', '-- Global Model --', '\033[0m')
    for key in model_global.keys():
        print(f'Best {key.upper()} : {model_global[key]}')
    print('\033[1m', global_score,'\033[0m')
    
    
    # Train clusters model
    model_clusters = {}
    results_clusters = {}
    results_global = {}
    for i in range(n_clusters):
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
        for crit in criterions:
            if crit not in model_clusters:
                model_clusters[crit] = []
            model_clusters[crit].append(model_i[crit])
            if crit not in results_clusters:
                results_clusters[crit] = []
            results_clusters[crit].append(results_i[crit])   
            
            # Compute cluster predictions based on global model
            predict_train_i[crit], predict_test_i[crit] = predict(
                model_global[crit],
                x_train_clusters[i],
                x_test_clusters[i]
            )

            # Compute MAE and R2 scores
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
    for crit in criterions:
        best_models[crit] = []
        best_results[crit] = []
    for i in range(n_clusters):
            
        if (results_global['r2'][i][1] > results_clusters['r2'][i][1]):
            best_models['r2'].append(model_global['r2'])
            best_results['r2'].append(results_global['r2'][i])
        else:
            best_models['r2'].append(model_clusters['r2'][i])
            best_results['r2'].append(results_clusters['r2'][i])

        if (results_global['mae'][i][3] < results_clusters['mae'][i][3]):
            best_models['mae'].append(model_global['mae'])
            best_results['mae'].append(results_global['mae'][i])
        else:
            best_models['mae'].append(model_clusters['mae'][i])
            best_results['mae'].append(results_clusters['mae'][i])

    print('-- Best Models --')
    for crit in criterions:
        for i in range(n_clusters):
            print(f'Best {crit} - Cluster {i} : {best_models[crit][i]}')
        
        print(pd.DataFrame(
            best_results[crit],
            index=[f'Best {crit.upper()} - Cluster {i}' for i in range(n_clusters)],
            columns=["Train R2",  "Test R2", "Train MAE", "Test MAE"]
        ))
    
    return best_models, global_score

def find_best_results(best_models, x_train, x_test, y_train, y_test):
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
    
    for i in range(n_clusters):
        idx_train = np.where(x_train["Cluster"]==i)
        x_train_clusters.append(x_train.iloc[idx_train].reset_index(drop=True))
        y_train_clusters.append(y_train.iloc[idx_train].reset_index(drop=True))

        idx_test = np.where(x_test["Cluster"]==i)
        x_test_clusters.append(x_test.iloc[idx_test].reset_index(drop=True))
        y_test_clusters.append(y_test.iloc[idx_test].reset_index(drop=True))
        
        predict_train_i = {}
        predict_test_i = {}
        for crit in criterions:
            # Compute cluster predictions
            predict_train_i[crit], predict_test_i[crit] = predict(
                best_models[crit][i],
                x_train_clusters[i],
                x_test_clusters[i]
            )
            predict_train_clusters[crit].append(predict_train_i[crit])
            predict_test_clusters[crit].append(predict_test_i[crit])
    
    y_train_merge = np.concatenate(y_train_clusters)
    y_test_merge = np.concatenate(y_test_clusters)
    
    predict_train_merge = {}
    predict_test_merge = {}
    train_mae = {}
    test_mae = {}
    train_r2 = {}
    test_r2 = {}
    
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
    best_models, results_global = find_best_models(
        models,
        x_train,
        x_test,
        y_train,
        y_test
    )
    optimized_results = find_best_results(
        best_models,
        x_train,
        x_test,
        y_train,
        y_test
    )
    return optimized_results, results_global, best_models

def KM_clustering(x_train, x_test, n_clusters=3):
    train = x_train.copy()
    test = x_test.copy()
    kmeanModel = KMeans(n_clusters=n_clusters).fit(train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    train['Cluster'] = kmeanModel.labels_
    test['Cluster'] = kmeanModel.predict(test.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    return train, test

def GMM_clustering(x_train, x_test, n_clusters=3):
    train = x_train.copy()
    test = x_test.copy()
    gmm = GaussianMixture(n_components=n_clusters).fit(train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    train['Cluster'] = gmm.predict(train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))
    test['Cluster'] = gmm.predict(test.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    return train, test

def standardize(x_train, x_test):
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)

    x_train_scaled = (x_train - train_mean)/train_std
    x_test_scaled = (x_test - train_mean)/train_std

    return x_train_scaled, x_test_scaled

def PLS_regression(x_train, x_test, y_train, y_test, n_components):
    pls = PLSRegression(n_components=n_components)
    pls.fit(x_train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'), y_train)

    new_x_train = pls.transform(x_train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))
    new_x_test = pls.transform(x_test.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))
    pls_train = pd.DataFrame(data = new_x_train, columns = ["class%02d" %i for i in range(1,n_components+1)])
    pls_test = pd.DataFrame(data = new_x_test, columns = ["class%02d" %i for i in range(1,n_components+1)])

    return pls_train, pls_test

def PCA_decomposition(x_train, x_test, y_train, y_test, n_components):
    pca = PCA(n_components=n_components)

    pc_train = pca.fit_transform(x_train.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))
    pc_test = pca.transform(x_test.drop(['Gender', 'Cluster'], axis=1, errors='ignore'))

    pc_train = pd.DataFrame(data = pc_train, columns = ["PC%02d" %i for i in range(1,pca.n_components_+1)])
    pc_test = pd.DataFrame(data = pc_test, columns = ["PC%02d" %i for i in range(1,pca.n_components_+1)])

    return pc_train, pc_test

def filter_mi_scores(x_train, x_test, y_train, threshold=0):
    mi_score_train = make_mi_scores(x_train.drop(['Cluster'], axis=1, errors='ignore'), y_train['Age'])

    keep_pls = (mi_score_train[mi_score_train > threshold].index.to_list())
    keep_pls.append('Cluster')

    x_train_filtered = x_train.copy()[keep_pls]
    x_test_filtered = x_test.copy()[keep_pls]

    return x_train_filtered, x_test_filtered

def filter_lasso(x_train, x_test, y_train, threshold=0):
    reg = LassoCV(cv=5, random_state=0, n_jobs=-1)
    reg.fit(x_train, y_train)
    coef = pd.Series(np.absolute(reg.coef_), index = x_train.columns)
    lasso_features = coef.iloc[coef.to_numpy() > threshold].index

    print("Lasso picked " + str(len(lasso_features)) + " variables and eliminated the other " +  str(len(coef)-len(lasso_features))  + " variables")

    x_train_filtered = x_train[lasso_features]
    x_test_filtered = x_test[lasso_features]

    return x_train_filtered, x_test_filtered