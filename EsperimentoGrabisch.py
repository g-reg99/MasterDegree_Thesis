"""
Created on Thu Dec 28 18:46:32 2023
@author: greg__99
"""

### IMPORTAZIONE DELLE LIBRERIE E DEL DATASET ###

#Importo le librerie che utilizzerò in questo codice
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import sys
sys.path.append('C:\\Users\\lorel\\Desktop\\GREGORIO\\TESI MAGISTRALE\\Script\\Articolo Grabisch')
from FunzioniShap import (exact_shap, kernell_shap, kadd_shap)
from FunzioniShap import (exact_group_shap, kernell_group_shap, kadd_group_shap)
from FunzioniShap import (exact_group_group_shap, kernell_group_group_shap, kadd_group_group_shap)
import os

#Importo il dataset da utilizzare, uso il dataset cinese. Lo divido in train e 
#test set, nel farlo setto un random state così da rendere la spartizione 
#replicabile. 
data = pd.read_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/Dataset/Dataset Cina/data_SHAPexperiment_IV0.05.csv')
data = data.drop(['Finanomalies97','Finanomalies112'], axis = 1)
X, y = data.iloc[:, 5:], data.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, 
                                                    train_size = 0.8)

### TRAINING DEI MODELLI DI MACHINE LEARNING ###

#Per prima cosa fitto un modello di regressione logistica sul train set.
model_log = sm.Logit(y_train, X_train)
model_log = model_log.fit()
#faccio delle prediction sul test set.
pred_log = model_log.predict(X_test)
for i in range(58):
    if pred_log.iloc[i] > 0.5:
        pred_log.iloc[i] = 1
    else:
        pred_log.iloc[i] = 0
#valuto la bontà classificatoria.
confusion_matrix(y_test, pred_log)
accuracy_score(y_test, pred_log)
recall_score(y_test, pred_log)
#l'accuracy è di circa il 60%, non molto buona.

#Costruisco i modelli di ML da interpretare, nello specifico si tratta di un 
#XGBoost (XGB) e di un Neural Network (NN). Per l'attività di parameter tuning
#si è stato scelto uno spazio parametrico abbastanza casuale e come metodi di
#selezione si usa un 10-fold CV e l'accuracy.
param_XGB = {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7, 10, 20, 50, 100],
               'subsample': [0.3, 0.5, 0.7], 'colsample_bytree': [0.3, 0.5, 0.7],
               'n_estimators': [10, 20, 30, 50, 100]}
random_state = np.random.RandomState(1)
model_XGB = RandomizedSearchCV(XGBClassifier(random_state = random_state), param_XGB, 
                               n_jobs = -1, random_state = random_state, 
                               scoring = 'accuracy', cv = 10)
model_XGB.fit(X_train, y_train)
pred_XGB = model_XGB.predict(X_test)
confusion_matrix(y_test, pred_XGB)
accuracy_score(y_test, pred_XGB)
recall_score(y_test, pred_XGB)

#l'accuracy è di circa il 67%, decisamente migliore rispetto al modello logistico.

param_NN = {"hidden_layer_sizes": [(50,100),(200,100),(30,10),(100,50)],
            "solver": ["lbfgs"], "learning_rate_init": [0.001, 0.01, 0.05]}
random_state = np.random.RandomState(10)
model_NN = GridSearchCV(MLPClassifier(max_iter=10**6, random_state=random_state), 
                        param_NN, n_jobs = -1, cv = 10, scoring = "accuracy")
model_NN.fit(X_train, y_train)
pred_NN = model_NN.predict(X_test)
confusion_matrix(y_test,pred_NN)
accuracy_score(y_test, pred_NN)
recall_score(y_test, pred_NN)
#l'accuracy è di circa il 65%, decisamente migliore rispetto al modello logistico.

#creo una dictionary con dentro entrambi i modelli
models = {'XGB': model_XGB, 'NN': model_NN}

### ESPERIMENTI: EXACT SHAP ###

#creo la cartella in cui salavre i risultati
os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiExactShap')

#calcolo il valore dell'Exact Shap per tutte le istanze nel test set, per ogni
#coviariata e per entrambi i modelli di ML analizzati. Misuro anche il tempo
#impiegato a calcolare tali valori e verifico la 'local accuracy'
for m in models:
    print('-----', m, '-----')
    results = exact_shap(X = X_test, X_train = X_train, model = models[m])

    result = pd.DataFrame(results['shap_matrix'], columns = X_train.columns)
    result['total_contri'] = result.sum(axis=1)
    result['mean_pred'] = models[m].predict_proba(X_train)[:,1].mean()
    result['sum'] = round(result.total_contri + result.mean_pred, 6)
    result['true_pred'] = models[m].predict_proba(X_test)[:,1]
    result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiExactShap/ExactShap%s.csv'%m, index=False, encoding='utf_8_sig')

    tim = (results['time_tot'], results['time_ind'])
    times = pd.DataFrame(tim)
    times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiExactShap/ExactShap%s_time.csv'%(m), index=False, encoding='utf_8_sig')

### ESPERIMENTI: KERNELL SHAP ###

os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellShap')

#definisco i valori di nm da testare e il numero di simulazioni da usare
nm = [50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 650, 800, 1000]
R = 100

#calcolo il valore del Kernell Shap per tutte le istanze nel test set, per 
#ogni gruppo di coviariate e per entrambi i modelli di ML analizzati. Ripeto la
#procedura su R simulazioni di dimensione nm e provo vari possibili valori di nm.
#Misuro  anche il tempo impiegato a calcolare tali valori.
for m in models:
    print('---------------', m, '---------------')
    os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellShap/KernellShap%s'%(m))
    for n in nm:
        print('----- dimention:', n, '-----')
        os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellShap/KernellShap%s/KernellShap%s_%s'%(m,m,n))
        risultati = kernell_shap(X = X_test, X_train = X_train, model = models[m], nm = n, R = R)
        for i in range(R):
            result = pd.DataFrame(risultati['shap_array'][:,:,i])
            result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellShap/KernellShap%s/KernellShap%s_%s/KernellShap%s_%s_%s.csv'%(m,m,n,m,n,i), index=False, encoding='utf_8_sig')
        tim = (risultati['time_set'], risultati['time_ind'])
        times = pd.DataFrame(tim)
        times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellShap/KernellShap%s/KernellShap%s_%s/KernellShap%s_%s_time.csv'%(m,m,n,m,n), index=False, encoding='utf_8_sig')

#per nm = 1024 utilizzo R = 1 perchè le  estrazioni sarebbero sempre le stesse
for m in models:
    print('---------------', m, '---------------')
    risultati = kernell_shap(X = X_test, X_train = X_train, model = models[m], nm = 1024, R = 1)
    result = pd.DataFrame(risultati['shap_array'][:,:,0])
    result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellShap/KernellShap%s/KernellShap%s_1024.csv'%(m,m), index=False, encoding='utf_8_sig')
    tim = (risultati['time_set'], risultati['time_ind'])
    times = pd.DataFrame(tim)
    times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellShap/KernellShap%s/KernellShap%s_1024_time.csv'%(m,m), index=False, encoding='utf_8_sig')


### ESPERIMENTI: 2-ADD e 3-ADD SHAP ###

#definisco i valori di additività da utilizzare
K = [2,3]

#calcolo il valore dei kadd Shap per tutte le istanze nel test set, per ogni 
#gruppo di coviariate, per entrambi i modelli di ML analizzati e per ogni valore
#di additività. Ripeto la procedura su R simulazioni di dimensione nm e provo 
#vari possibili valori di nm. Misuro  anche il tempo impiegato a calcolare tali 
#valori.
for k in K:
    print('-------------------- additivity:', k, '--------------------')
    os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddShap'%k)
    for m in models: 
        print('---------------', m, '---------------')
        os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddShap/%saddShap%s'%(k,k,m))
        for n in nm:
            print('----- dimention:', n, '-----')
            os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddShap/%saddShap%s/%saddShap%s_%s'%(k,k,m,k,m,n))
            risultati = kadd_shap(X = X_test, X_train = X_train, model = models[m], k = k, nm = n, R = R)
            for i in range(R):
                result = pd.DataFrame(risultati['shap_array'][:,0:57,i])
                result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddShap/%saddShap%s/%saddShap%s_%s/%saddShap%s_%s_%s.csv'%(k,k,m,k,m,n,k,m,n,i), index=False, encoding='utf_8_sig')
            tim = (risultati['time_set'], risultati['time_ind'])
            times = pd.DataFrame(tim)
            times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddShap/%saddShap%s/%saddShap%s_%s/%saddShap%s_%s_time.csv'%(k,k,m,k,m,n,k,m,n), index=False, encoding='utf_8_sig')

### ESPERIMENTI: GROUP EXACT SHAP ###

os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiExactGroupShap')

#definisco i gruppi di covariate
gruppi = [[0,1], [2,3,4], [5,6], [7,8], [9]]

#calcolo il valore dell'Exact Group Shap per tutte le istanze nel test set, per 
#ogni gruppo di coviariate e per entrambi i modelli di ML analizzati. Misuro 
#anche il tempo impiegato a calcolare tali valori e verifico la 'local accuracy'
for m in models:
    print('-----', m, '-----')
    results = exact_group_shap(X = X_test, X_train = X_train, model = models[m], 
                               groups = gruppi)

    result = pd.DataFrame(results['shap_matrix'], columns = ['operation', 'profitability', 'solvency', 'leverage', 'growth'])
    result['total_contri'] = result.sum(axis=1)
    result['mean_pred'] = models[m].predict_proba(X_train)[:,1].mean()
    result['sum'] = round(result.total_contri + result.mean_pred, 6)
    result['true_pred'] = models[m].predict_proba(X_test)[:,1]
    result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiExactGroupShap/ExactGroupShap%s.csv'%m, index=False, encoding='utf_8_sig')

    tim = (results['time_tot'], results['time_ind'])
    times = pd.DataFrame(tim)
    times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiExactGroupShap/ExactGroupShap%s_time.csv'%(m), index=False, encoding='utf_8_sig')

### ESPERIMENTI: GROUP KERNELL SHAP ###

os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupShap')

#definisco i valori di 'nm' da utilizzare e il numero di simulazioni
nm = [60, 75, 100, 150, 200, 250, 300, 350, 400, 500, 650, 800, 1000]
R = 100

#calcolo il valore del Kernell Group Shap per tutte le istanze nel test set, per 
#ogni gruppo di coviariate e per entrambi i modelli di ML analizzati. Ripeto la
#procedura su R simulazioni di dimensione nm e provo vari possibili valori di nm.
#Misuro  anche il tempo impiegato a calcolare tali valori.
for m in models:
    print('---------------', m, '---------------')
    os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupShap/KernellGroupShap%s'%(m))
    for n in nm:
        print('----- dimention:', n, '-----')
        os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupShap/KernellGroupShap%s/KernellGroupShap%s_%s'%(m,m,n))
        risultati = kernell_group_shap(X = X_test, X_train = X_train, model = models[m], groups = gruppi, nm = n, R = R)
        for i in range(R):
            result = pd.DataFrame(risultati['shap_array'][:,:,i])
            result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupShap/KernellGroupShap%s/KernellGroupShap%s_%s/KernellGroupShap%s_%s_%s.csv'%(m,m,n,m,n,i), index=False, encoding='utf_8_sig')
        tim = (risultati['time_set'], risultati['time_ind'])
        times = pd.DataFrame(tim)
        times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupShap/KernellGroupShap%s/KernellGroupShap%s_%s/KernellGroupShap%s_%s_time.csv'%(m,m,n,m,n), index=False, encoding='utf_8_sig')

#per nm = 1024 utilizzo R = 1 perchè le  estrazioni sarebbero sempre le stesse.
for m in models:
    print('---------------', m, '---------------')
    risultati = kernell_group_shap(X = X_test, X_train = X_train, model = models[m], groups = gruppi, nm = 1024, R = 1)
    result = pd.DataFrame(risultati['shap_array'][:,:,0])
    result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupShap/KernellGroupShap%s/KernellGroupShap%s_1024.csv'%(m,m), index=False, encoding='utf_8_sig')
    tim = (risultati['time_set'], risultati['time_ind'])
    times = pd.DataFrame(tim)
    times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupShap/KernellGroupShap%s/KernellGroupShap%s_1024_time.csv'%(m,m), index=False, encoding='utf_8_sig')

### ESPERIMENTI: GROUP KADD SHAP ###

#definisco i gradi di additività da utilizzare
K = [2,3]

#calcolo il valore dei kadd group Shap per tutte le istanze nel test set, per ogni 
#gruppo di coviariate, per entrambi i modelli di ML analizzati e per ogni valore
#di additività. Ripeto la procedura su R simulazioni di dimensione nm e provo 
#vari possibili valori di nm. Misuro  anche il tempo impiegato a calcolare tali 
#valori.
for k in K:
    print('-------------------- additivity:', k, '--------------------')
    os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupShap'%k)
    for m in models: 
        print('---------------', m, '---------------')
        os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupShap/%saddGroupShap%s'%(k,k,m))
        for n in nm:
            print('----- dimention:', n, '-----')
            os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupShap/%saddGroupShap%s/%saddGroupShap%s_%s'%(k,k,m,k,m,n))
            risultati = kadd_group_shap(X = X_test, X_train = X_train, model = models[m], k = k, groups = gruppi, nm = n, R = R)
            for i in range(R):
                result = pd.DataFrame(risultati['shap_array'][:,:,i])
                result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupShap/%saddGroupShap%s/%saddGroupShap%s_%s/%saddGroupShap%s_%s_%s.csv'%(k,k,m,k,m,n,k,m,n,i), index=False, encoding='utf_8_sig')
            tim = (risultati['time_set'], risultati['time_ind'])
            times = pd.DataFrame(tim)
            times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupShap/%saddGroupShap%s/%saddGroupShap%s_%s/%saddGroupShap%s_%s_time.csv'%(k,k,m,k,m,n,k,m,n), index=False, encoding='utf_8_sig')

### ESPEROMENTI: GROUP GROUP EXACT SHAP ###

os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiExactGroupGroupShap')

#definisco i gruppi di covariate
gruppi = [[0,1], [2,3,4], [5,6], [7,8], [9]]

#calcolo il valore dell'Exact Group Group Shap per tutte le istanze nel test set, 
#per ogni gruppo di coviariate e per entrambi i modelli di ML analizzati. Misuro 
#anche il tempo impiegato a calcolare tali valori e verifico la 'local accuracy'
for m in models:
    print('-----', m, '-----')
    results = exact_group_group_shap(X = X_test, X_train = X_train, model = models[m], 
                                     groups = gruppi)

    result = pd.DataFrame(results['shap_matrix'], columns = ['operation', 'profitability', 'solvency', 'leverage', 'growth'])
    result['total_contri'] = result.sum(axis=1)
    result['mean_pred'] = models[m].predict_proba(X_train)[:,1].mean()
    result['sum'] = round(result.total_contri + result.mean_pred, 6)
    result['true_pred'] = models[m].predict_proba(X_test)[:,1]
    result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiExactGroupGroupShap/ExactGroupGroupShap%s.csv'%m, index=False, encoding='utf_8_sig')

    tim = (results['time_tot'], results['time_ind'])
    times = pd.DataFrame(tim)
    times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiExactGroupGroupShap/ExactGroupGroupShap%s_time.csv'%(m), index=False, encoding='utf_8_sig')

### ESEPRIMENTI: GROUP GROUP KERNELL SHAP ###

os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupGroupShap')

#definisco i valori di 'nm' da utilizzare e il numero di simulazioni
nm = [13, 15, 17, 20, 22, 25, 27, 30]
R = 100

#calcolo il valore del Kernell Group Group Shap per tutte le istanze nel test set, 
#per ogni gruppo di coviariate e per entrambi i modelli di ML analizzati. Ripeto la
#procedura su R simulazioni di dimensione nm e provo vari possibili valori di nm.
#Misuro  anche il tempo impiegato a calcolare tali valori.
for m in models:
    print('---------------', m, '---------------')
    for n in nm:
        print('----- dimention:', n, '-----')
        os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupGroupShap/KernellGroupGroupShap%s/KernellGroupGroupShap%s_%s'%(m,m,n))
        risultati = kernell_group_group_shap(X = X_test, X_train = X_train, model = models[m], groups = gruppi, nm = n, R = R)
        for i in range(R):
            result = pd.DataFrame(risultati['shap_array'][:,:,i])
            result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupGroupShap/KernellGroupGroupShap%s/KernellGroupGroupShap%s_%s/KernellGroupGroupShap%s_%s_%s.csv'%(m,m,n,m,n,i), index=False, encoding='utf_8_sig')
        tim = (risultati['time_set'], risultati['time_ind'])
        times = pd.DataFrame(tim)
        times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupGroupShap/KernellGroupGroupShap%s/KernellGroupGroupShap%s_%s/KernellGroupGroupShap%s_%s_time.csv'%(m,m,n,m,n), index=False, encoding='utf_8_sig')

#per nm = 32 utilizzo R = 1 perchè le  estrazioni sarebbero sempre le stesse.
for m in models:
    print('---------------', m, '---------------')
    risultati = kernell_group_group_shap(X = X_test, X_train = X_train, model = models[m], groups = gruppi, nm = 32, R = 1)
    result = pd.DataFrame(risultati['shap_array'][:,:,0])
    result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupGroupShap/KernellGroupGroupShap%s/KernellGroupGroupShap%s_32.csv'%(m,m), index=False, encoding='utf_8_sig')
    tim = (risultati['time_set'], risultati['time_ind'])
    times = pd.DataFrame(tim)
    times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/RisultatiKernellGroupGroupShap/KernellGroupGroupShap%s/KernellGroupGroupShap%s_32_time.csv'%(m,m), index=False, encoding='utf_8_sig')

### ESPERIMENTI: GROUP GROUP KADD SHAP ###

#definisco i gradi di additività da utilizzare e i valori di 'nm'
K = [2,3]
nm = [20, 22, 25, 27, 30, 32]

#calcolo il valore dei kadd group group Shap per tutte le istanze nel test set, 
#per ogni gruppo di coviariate, per entrambi i modelli di ML analizzati e per 
#ogni valore di additività. Ripeto la procedura su R simulazioni di dimensione 
#nm e provo vari possibili valori di nm. Misuro  anche il tempo impiegato a 
#calcolare tali valori.
for k in K:
    print('-------------------- additivity:', k, '--------------------')
    os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupGroupShap'%k)
    for m in models: 
        print('---------------', m, '---------------')
        os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupGroupShap/%saddGroupGroupShap%s'%(k,k,m))
        for n in nm:
            print('----- dimention:', n, '-----')
            os.mkdir('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupGroupShap/%saddGroupGroupShap%s/%saddGroupGroupShap%s_%s'%(k,k,m,k,m,n))
            risultati = kadd_group_group_shap(X = X_test, X_train = X_train, model = models[m], k = k, groups = gruppi, nm = n, R = R)
            for i in range(R):
                result = pd.DataFrame(risultati['shap_array'][:,0:17,i])
                result.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupGroupShap/%saddGroupGroupShap%s/%saddGroupGroupShap%s_%s/%saddGroupGroupShap%s_%s_%s.csv'%(k,k,m,k,m,n,k,m,n,i), index=False, encoding='utf_8_sig')
            tim = (risultati['time_set'], risultati['time_ind'])
            times = pd.DataFrame(tim)
            times.to_csv('/Users/lorel/Desktop/GREGORIO/TESI MAGISTRALE/script/Articolo Grabisch/Risultati%saddGroupGroupShap/%saddGroupGroupShap%s/%saddGroupGroupShap%s_%s/%saddGroupGroupShap%s_%s_time.csv'%(k,k,m,k,m,n,k,m,n), index=False, encoding='utf_8_sig')
