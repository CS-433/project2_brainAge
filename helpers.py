import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression





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
    predict_train = model.predict(x_train)
    predict_test = model.predict(x_test)
    
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
    train_mae = mean_absolute_error(y_train, predict_train)
    test_mae = mean_absolute_error(y_test, predict_test)
    
    return train_mae, test_mae

def compute_r2(y_train, y_test, predict_train, predict_test):
    train_r2 = r2_score(y_train, predict_train)
    test_r2 = r2_score(y_test, predict_test)
    
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

def train_model(title, gs, x_train, x_test, y_train, y_test, plot):

    # Fit using grid search
    gs.fit(x_train.drop(['Cluster'], axis=1, errors='ignore'), y_train['Age'])

    # Compute predictions
    predict_train, predict_test = predict(
        gs.best_estimator_,
        x_train.drop(['Cluster'], axis=1, errors='ignore'),
        x_test.drop(['Cluster'], axis=1, errors='ignore')
    )
      
    # Compute MAE and R2 scores
    train_mae, test_mae = compute_mae(y_train['Age'], y_test['Age'], predict_train, predict_test)
    train_r2, test_r2 = compute_r2(y_train['Age'], y_test['Age'], predict_train, predict_test)
    
    if (plot == True):
        plot_results(title, x_train, x_test, y_train, y_test, predict_train, predict_test)

    return gs.best_score_, train_r2, test_r2, train_mae, test_mae

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
        )
        
    return results

def train_cluster(i, x_train, x_test, y_train, y_test, models, plot=True):
    idx_train = np.where(x_train["Cluster"]==i)
    x_train_cluster = x_train.iloc[idx_train]
    y_train_cluster = y_train.iloc[idx_train]

    idx_test = np.where(x_test["Cluster"]==i)
    x_test_cluster = x_test.iloc[idx_test]
    y_test_cluster = y_test.iloc[idx_test]
    
    results = train_all(x_train_cluster, x_test_cluster, y_train_cluster, y_test_cluster, models, plot, 'Cluster ' + str(i))
    
    return results