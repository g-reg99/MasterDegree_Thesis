"""
Created on Wed Jan  3 19:08:55 2024

@author: greg_99
"""

import numpy as np
import itertools
import math
import time
from scipy.special import comb
from scipy.special import bernoulli

def exact_shap(X, X_train, model):
    """
    Calcola l'exact SHAP per un vettore di istanze. Misura anche il tempo
    medio impiegato per valutare ogni istanza e il tempo impegato per valutare
    l'intero vettore di istanze.
    ----------
    PARAMETERS:
    X = desing matrix delle istanze di interesse.
    X_train  = desing matrix delle istanze usate per trainare il modello.
    model = modello da interpretare.
    ------
    RETURN:
    shap_matrix = matrice con gli exact shap per ogni istanza e features.
    time_tot = tempo impiegato per il calcolo dell'intera matrice.
    time_ind = tempo medio impiegato per valutare la singola istanza.
    """
    instances = range(len(X))    
    features = X.columns
    X = np.array(X)
    shap = np.ones([len(X), len(features)])
    
    start = time.time()
    
    #creo tutte le possibili combo tra le features
    all_subsets = [] 
    for k in range(len(features), -1, -1):
        for element in itertools.combinations(range(len(features)), k):
            all_subsets.append(list(element))
            
    #assegno un peso a ogni combinazione
    omega = []
    for l in all_subsets[1:]:
        omegal = (math.factorial(len(l)) * math.factorial(len(features) - len(l) - 1)) / math.factorial(len(features))
        omega.append(omegal)
    
    #calcolo gli exact shap per una istanza
    for i in range(len(instances)):
        print('instance', i) 
        temp = np.tile(X[i], (len(X_train), 1))
        
        #calcolo i valori attesi legati all'istanza i-esima per ogni combinazione
        all_val = []
        for l in all_subsets:
            mat = np.array(X_train)
            mat[:, l] = temp[:, l]
            val = list(model.predict_proba(mat)[:,1])
            valS = np.array(val)
            valS = valS.mean()
            all_val.append(valS)
            
        #calcolo l'exact shap per la j-esima features
        for j in range(len(features)):
            Phi = 0
            for l in range(len(all_subsets)):
                if j not in all_subsets[l]:
                    valore = all_val[l]
                    peso = omega[l - 1]
                    lj = all_subsets[l] + [j]; lj.sort()
                    indice = all_subsets.index(lj)
                    valorej = all_val[indice]
                    phi = peso * (valorej - valore)
                    Phi += phi
            
            shap[i,j] = Phi
                
    end = time.time()
    time_tot = round(end - start, 8)
    time_ind = round(time_tot/len(X), 8)
        
    results = {'shap_matrix': shap, 'time_tot': time_tot, 'time_ind': time_ind}
    return results

def kernell_shap(X, X_train, model, nm, R):
    """
    Calcola i kernell SHAP per un vettore di istanze utilizzando 'nm' coalizioni
    estratte dal power set di M. Ripete la procedura per R differenti estrazioni
    di uguale dimensione 'nm'. Misura anche il tempo medio necessario per fare
    questi calcoli.
    ----------
    PARAMETERS:
    X = desing matrix delle istanze di interesse.
    X_train  = desing matrix delle istanze usate per trainare il modello.
    model = modello da interpretare.
    nm = numero di coalizioni utilizzate nel calcolo.
    R = numero di estrazioni fatte.
    ----------
    RETURN:
    shap_array = array contenente una matrice di kernell shap per ogni estrazione
    r-esima. Una singola matrice contiene i kernell shap per ogni features e 
    per ogni istanza.
    time_ind = tempo mediamente necessario per calcolare i kernell shap per 
    una singola istanza.
    time_set = tempo mediamente necessario per calcolare una matrice.
    """
    features = X.columns
    instances = range(len(X))
    X = np.array(X)
       
    #costruisco il power set di M in forma di lista
    PL = []
    for k in range(len(features), -1, -1):
        for element in itertools.combinations(range(len(features)), k):
            PL.append(list(element))

    #costruisco il power set di M come matrice di 0,1
    PM = np.zeros([2**len(features), len(features)]) 
    for l in range(2**len(features)):
        PM[l, PL[l]] = 1
     
    #costruisco i pesi che ogni coalzione avrà nella estrazione randomica
    weights_PM = [] 
    for m in range(2**len(features)):
        w = (len(features) - 1)/(comb(len(features), PM[m].sum())*PM[m].sum()*(len(features) - PM[m].sum()))
        weights_PM.append(w) 
    weights_PM[0] = 10**6
    weights_PM[-1] = 10**6
    #costruisco i pesi normalizzati
    weights_PM_norm = weights_PM/np.array(weights_PM).sum()
    
    #definisco il valore massimo di nm (che è il numero totale di coalizioni)
    nm_max = len(PL)
    
    if nm <= nm_max:
        
        #creo un array dove andranno i kernell SHAP di ogni sumulazione r-esima
        phi_array = np.ones([len(X), len(features) + 1, R])
    
        start = time.time()
        for r in range(R):
            print('simulation:', r)
            SM = np.ones([nm, len(features)])
            SL = []
            weights = np.ones([nm])
            I = range(2**len(features))
            #estraggo le coalizioni che userò nella valutazione
            np.random.seed(r)
            estrazioni = np.random.choice(I, size = nm, replace = False, p = weights_PM_norm) 
        
            for s in range(nm):
                SM[s] = PM[estrazioni[s]]
                weights[s] = weights_PM[estrazioni[s]]
                sl = PL[estrazioni[s]]
                SL.append(sl)
        
            #costruisco gli elementi da usare nel calcolo di 'S'
            Z = np.concatenate((np.ones([nm,1]), SM), axis = 1)
            W = np.diag(weights)
            #calcolo S (che è lo stesso per ogni istanza)
            S = np.linalg.inv(Z.T @ W @ Z) @ Z.T @ W
        
            #creo una matrice dove andranno i phi relativi a ogni istanza
            phi_matrix = np.ones([len(X), len(features) + 1])
        
            for i in range(len(instances)):
                #costruisco il vettore 'f' di valori attesi
                f = []
                temp = np.tile(X[i], (len(X_train), 1))
            
                for j in SL:
                    mat = np.array(X_train)
                    mat[:, j] = temp[:, j]
                    val = list(model.predict_proba(mat)[:,1])
                    valS = np.array(val)
                    valS = valS.mean()
                    f.append(valS)
                
                #calcolo i kernell SHAP per l'istanza i-esima (compreso phi_0)
                phi = S @ f
                phi_matrix[i] = phi
        
            phi_array[:,:,r] = phi_matrix
        
        end = time.time()
        #misuro il tempo impiegato per calcolare i kernell shap per ogni
        #istanza e per tutte le R simulazioni
        time_cons = end - start
        #calcolo il tempo che mediamente è stato necessatio per calcolare
        #i kernell shap di tutte le istanze per una singola simulazione
        tempo_medio_tot = round(time_cons/R,8)
        #calcolo il tempo medio per una singola istanza
        tempo_medio_ind = round(tempo_medio_tot/len(X),8)
    
        results = {'shap_array': phi_array, 'time_ind': tempo_medio_ind, 'time_set': tempo_medio_tot}
        return results
    
    else:
        print('"nm" supera il valore massimo. Valore massimo di "nm":', nm_max)

def kadd_shap(X, X_train, model, k, nm, R):
    """
    Calcola i kadd SHAP per un vettore di istanze utilizzando 'nm' coalizioni
    estratte dal power set di M. Ripete la procedura per R differenti estrazioni
    di uguale dimensione 'nm'. Misura anche il tempo medio necessario per fare
    questi calcoli.
    ----------
    PARAMETERS:
    X = desing matrix delle istanze di interesse.
    X_train  = desing matrix delle istanze usate per trainare il modello.
    model = modello da interpretare.
    k = grado di additività del modello.
    nm = numero di coalizioni utilizzate nel calcolo.
    R = numero di estrazioni fatte.
    ----------
    RETURN:
    shap_array = array contenente una matrice di kadd shap per ogni estrazione
    r-esima. Una singola matrice contiene i kadd shap per ogni features, per ogni
    interazione di grado miore o uguale a k e per ogni istanza.
    time_ind = tempo mediamente necessario per calcolare i kadd shap per 
    una singola istanza.
    time_set = tempo mediamente necessatio per calcolare una matrice.
    """
    features = X.columns
    instances = range(len(X))
    X = np.array(X)
       
    #costruisco il power set di M, ma in questo caso lo costruisco partendo da
    #j = 0 in modo da avere le interazioni di secondo grado nelle prime posizioni
    PL = []
    for j in range(len(features) + 1):
        for element in itertools.combinations(range(len(features)), j):
            PL.append(list(element))
    
    #costruisco il set di combinazioni con cardinalità inferiore a k
    DL = []
    for el in PL:
        if len(el) <= k:
            DL.append(el)
    
    #calcolo il numero di parametri e il numero di Bernoulli corrispondenti 
    #al grado di additività dato in input
    n_par = len(DL)
    bern = bernoulli(k)

    #ricavo PL e DL in forma di vettori binari
    PM = np.zeros([2**len(features), len(features)]) 
    for l in range(2**len(features)):
        PM[l, PL[l]] = 1
    
    DM = np.zeros([n_par, len(features)]) 
    for ll in range(n_par):
        DM[ll, DL[ll]] = 1
    
    #calcolo i pesi da usare nell'estrazione delle coalizioni esattamente come
    #nel kernell shap
    weights_PM = [] 
    for m in range(2**len(features)):
        w = (len(features) - 1)/(comb(len(features), PM[m].sum())*PM[m].sum()*(len(features) - PM[m].sum()))
        weights_PM.append(w) 
    weights_PM[0] = 10**6
    weights_PM[-1] = 10**6
    #costruisco i pesi normalizzati
    weights_PM_norm = weights_PM/np.array(weights_PM).sum()
   
    #costruisco i pesi da usare nella procedura di ottimizzazione
    weights_k = np.ones(2**len(features))
    weights_k[0] = 10**6
    weights_k[-1] = 10**6
    
    nm_max = len(PL)
    
    if nm <= nm_max:
        #costruisco l'array dove andrò a mettere i risultati
        phi_array = np.ones([len(X), n_par, R])
    
        start = time.time()
        for r in range(R):
            print('simulation:', r)
            SM = np.ones([nm, len(features)])
            SL = []
            weights = np.ones([nm])
            I = range(2**len(features))
            #estraggo le coalizioni che userò nella valutazione
            np.random.seed(r)
            estrazioni = np.random.choice(I, size = nm, replace = False, p = weights_PM_norm) 
        
            for s in range(nm):
                SM[s] = PM[estrazioni[s]]
                weights[s] = weights_k[estrazioni[s]]
                sl = PL[estrazioni[s]]
                SL.append(sl)
        
            #costruisco la Trasformation Matrix
            T = np.ones([nm, n_par])
        
            for m in range(nm):
                for n in range(n_par):
                    Dcard = int(sum(DM[n,:]))
                    ADcard = int(sum(DM[n,:]*SM[m,:]))
                    gamma = 0
                    for lll in range(ADcard + 1):
                        gamma += comb(ADcard, lll) * bern[Dcard - lll]
                    T[m,n] = gamma
        
            #costruisco la matrice dei pesi
            W = np.diag(weights)
            #costruisco la matrice S che sarà costante lungo ogni istanza
            S = np.linalg.inv(T.T @ W @ T) @ T.T @ W
        
            phi_matrix = np.ones([len(X), n_par])
        
            for i in range(len(instances)):
                #costruisco il vettore 'f' di valori attesi
                f = []
                temp = np.tile(X[i], (len(X_train), 1))
                #calcolo phi0 come predzione media sul train set
                phi0 = model.predict_proba(X_train)[:,1].mean()
            
                for j in SL:
                    mat = np.array(X_train)
                    mat[:, j] = temp[:, j]
                    val = list(model.predict_proba(mat)[:,1])
                    valS = np.array(val)
                    valS = valS.mean()
                    f.append(valS)
                
                f_bar = f - phi0
                phi = S @ f_bar
                phi_matrix[i] = phi

            phi_array[:,:,r] = phi_matrix
        
        end = time.time()
        #misuro il tempo necessario per costruire l'intero array
        time_cons = end - start
        #calcolo il tempo medio per costruira una matrice
        tempo_medio_tot = round(time_cons/R,8)
        #calcolo il tempo medio per costruire una iterazione
        tempo_medio_ind = round(tempo_medio_tot/len(X),8)
    
        results = {'shap_array': phi_array, 'time_ind': tempo_medio_ind, 'time_set': tempo_medio_tot}
        return results
    
    else:
        print('"nm" supera il valore massimo. Valore massimo di "nm":', nm_max)

def exact_group_shap(X, X_train, model, groups):
    """
    Calcola l'exact group SHAP per un vettore di istanze. Misura anche il tempo
    medio impiegato per valutare ogni istanza.
    ----------
    PARAMETERS:
    X = desing matrix delle istanze di interesse.
    X_train  = desing matrix delle istanze usate per trainare il modello.
    model = modello da interpretare.
    groups = i gruppi di covariate di cui di vuole valutare l'impatto nelle
    predizioni.
    ------
    RETURN:
    shap_matrix = matrice con gli exact group shap per ogni istanza e features.
    time_tot = tempo impiegato per il calcolo dell'intera matrice.
    time_ind = tempo medio impiegato per valutare la singola istanza.
    """
    instances = range(len(X))
    features = groups
    #dando le features in input come gruppo abbiamo bisogno di disaggregarle
    features0 = [m for n in features for m in n]
    X = np.array(X)
    shap = np.ones([len(X), len(features)])
    
    start = time.time()
    
    #creo l'insieme di tutte le combinazioni ottenute fissando un gruppo alla
    #volta e trattando le altre variabili individualmente.
    all_subsets = []
    for j in range(len(features)):
        exclusive = features.copy()
        del(exclusive[j])
        exclusive = [m for n in exclusive for m in n]
        for k in range(len(exclusive), -1, -1):
            for element in itertools.combinations(exclusive, k):
                all_subsets.append(list(element))
                all_subsets.append(list(element) + features[j])
                
    for el in all_subsets:
        el = el.sort()
    
    #elimino i doppioni
    sub = []
    for el in all_subsets:
        if sub.count(el) < 1:
            sub.append(el)
 
    #calcolo gli exact group shap per l' i-esima istanza
    for i in range(len(instances)): 
        print('instance:', i)
        temp = np.tile(X[i], (len(X_train), 1))
        
        all_val = []
        for l in sub:
            mat = np.array(X_train)
            mat[:, l] = temp[:, l]
            val = list(model.predict_proba(mat)[:,1])
            valS = np.array(val)
            valS = valS.mean()
            all_val.append(valS)
        
        #calcolo l'exact group shap relativo alla j-esima features
        for j in range(len(features)):
            Phi = 0
            for l in range(len(sub)):
                num = 0
                for element in features[j]:
                    if element not in sub[l]:
                        num += 1
                if num == len(features[j]):
                    valore = all_val[l]
                    peso = (math.factorial(len(sub[l])) * math.factorial(len(features0) - len(sub[l]) - len(features[j]))) / math.factorial(len(features0)-len(features[j])+1)
                    lj = sub[l] + features[j]; lj.sort()
                    indice = sub.index(lj)
                    valorej = all_val[indice]
                    phi = peso * (valorej - valore)
                    Phi += phi
                
            shap[i, j] = Phi
        
    end = time.time()
    #tempo per calcolare l'intera matrice
    time_tot = round(end - start, 8)
    #tempo medio per calcolare una singola istanza
    time_ind = round(time_tot/len(X), 8)
        
    results = {'shap_matrix': shap, 'time_tot': time_tot, 'time_ind': time_ind}
    return results

def kernell_group_shap(X, X_train, model, groups, nm, R):
    """
    Calcola i kernell Group SHAP per un vettore di istanze utilizzando 'nm' 
    coalizioni estratte dal power set di M. Ripete la procedura per R differenti 
    estrazioni di uguale dimensione 'nm'. Misura anche il tempo medio necessario 
    per fare questi calcoli.
    ----------
    PARAMETERS:
    X = desing matrix delle istanze di interesse.
    X_train  = desing matrix delle istanze usate per trainare il modello.
    model = modello da interpretare.
    groups = lista dei gruppi da analizzare.
    nm = numero di coalizioni utilizzate nel calcolo.
    R = numero di estrazioni fatte.
    ----------
    RETURN:
    shap_array = array contenente una matrice di kernell group shap per ogni 
    estrazione r-esima. Una singola matrice contiene i kernell group shap per 
    ogni features e per ogni istanza.
    time_ind = tempo mediamente necessario per calcolare i kernell group shap per 
    una singola istanza.
    time_set = tempo mediamente necessario per calcolare una matrice.
    """
    features = groups
    features0 = [m for n in features for m in n]
    instances = range(len(X))
    X = np.array(X)
 
    #creo l'insieme di tutte le combinazioni ottenute fissando un gruppo alla
    #volta e trattando le altre variabili individualmente.
    all_subsets = []
    for j in range(len(features)):
        exclusive = features.copy()
        del(exclusive[j])
        exclusive = [m for n in exclusive for m in n]
        for k in range(len(exclusive), -1, -1):
            for element in itertools.combinations(exclusive, k):
                all_subsets.append(list(element))
                all_subsets.append(list(element) + features[j])
                
    for el in all_subsets:
        el = el.sort()
         
    PL = []
    for el in all_subsets:
        if PL.count(el) < 1:
            PL.append(el)

    #converto la lista di coalizioni in vettori binari
    PM = np.zeros([len(PL), len(features0)]) 
    for l in range(len(PL)):
        PM[l, PL[l]] = 1
        
    #calcolo i pesi da usare nell'estrazione delle coalizioni
    weights_PM = [] 
    for m in range(len(PL)):
        w = (len(features0) - 1)/(comb(len(features0), PM[m].sum())*PM[m].sum()*(len(features0) - PM[m].sum()))
        weights_PM.append(w)
    for m in range(len(weights_PM)):
        if weights_PM[m] == float('inf'):
            weights_PM[m] = 10**6
    weights_PM_norm = weights_PM/np.array(weights_PM).sum()
    
    #definisco il numero massimo di estrazioni effettuabili
    nm_max = len(PL)
    
    if nm <= nm_max:
        
        phi_array = np.ones([len(X), len(features), R])
        
        start = time.time()
        #effettuo un'estrazione
        for r in range(R):
            print('simulation:', r)
            SM = np.ones([nm, len(features0)])
            SL = []
            weights = np.ones([nm])
            I = range(len(PL))
            np.random.seed(r) 
            estrazioni = np.random.choice(I, size = nm, replace = False, p = weights_PM_norm) 
    
            for s in range(nm):
                SM[s] = PM[estrazioni[s]]
                weights[s] = weights_PM[estrazioni[s]]
                sl = PL[estrazioni[s]]
                SL.append(sl)
    
            #calcolo i valori attesi per ogni estrazione e per ogni istanza
            f = np.ones([len(instances), nm])
        
            for i in range(len(instances)):
                ff = []
                temp = np.tile(X[i], (len(X_train), 1)) 
                
                for l in SL:
                    mat = np.array(X_train)
                    mat[:, l] = temp[:, l]
                    val = list(model.predict_proba(mat)[:,1])
                    valS = np.array(val)
                    valS = valS.mean()
                    ff.append(valS)
        
                f[i,:] = ff
    
            phi_matrix = np.ones([len(X), len(features)])
            
            #calcolo i kernell group shap relativi alla features j-esima e per
            #ogni istanza
            for j in range(len(features)):
                Z = np.ones([nm, 1])
                wj = np.ones([nm, 1])
                K = len(features0) - len(features[j]) + 1
                
                for n in range(nm):
                    
                    if SM[n, features[j]].sum() == len(features[j]):
                        Z[n] = 1
                        omega = SM[n].sum() - len(features[j]) + 1
                        wj[n] = (K - 1)/(comb(K, omega) * omega * (K - omega))
                    
                    elif SM[n, features[j]].sum() == 0:
                        Z[n] = 0
                        wj[n] = (K - 1)/(comb(K, SM[n].sum()) * (SM[n].sum()) * (K - SM[n].sum()))
                        
                    else:
                        Z[n] = 0
                        wj[n] = 0
        
                    for m in range(len(wj)):
                        if wj[m] == float('inf'):
                            wj[m] = 10**6
                
                ZJ = np.concatenate((np.ones([nm, 1]), SM[:, 0:features[j][0]], Z, SM[:, features[j][-1] + 1:len(features0)]), axis=1)
                W = wj * np.identity(nm)
                S = np.linalg.inv(ZJ.T @ W @ ZJ) @ ZJ.T @ W

                phi = S @ f.T
                pos = features[j][0] + 1
                phi_matrix[:, j] = phi[pos, :]
                
            phi_array[:,:,r] = phi_matrix
        
        end = time.time()
        #tempo impiegato per la costruzione di un array
        time_cons = end - start
        #tempo medio impiegato per la costruzione di una matrice
        tempo_medio_tot = round(time_cons/R,8)
        #tempo medio impiegato per valutare una istanza
        tempo_medio_ind = round(tempo_medio_tot/len(X),8)
    
        results = {'shap_array': phi_array, 'time_ind': tempo_medio_ind, 'time_set': tempo_medio_tot}
        return results
    
    else:
        print('"nm" supera il valore massimo. Valore massimo di "nm":', nm_max)

def kadd_group_shap(X, X_train, model, k, groups, nm, R):
    """
    Calcola i kadd group SHAP per un vettore di istanze utilizzando 'nm' coalizioni
    estratte dal power set di M. Ripete la procedura per R differenti estrazioni
    di uguale dimensione 'nm'. Misura anche il tempo medio necessario per fare
    questi calcoli.
    ----------
    PARAMETERS:
    X = desing matrix delle istanze di interesse.
    X_train  = desing matrix delle istanze usate per trainare il modello.
    model = modello da interpretare.
    k = grado di additività del modello.
    groups = lista dei gruppi da analizzare.
    nm = numero di coalizioni utilizzate nel calcolo.
    R = numero di estrazioni fatte.
    ----------
    RETURN:
    shap_array = array contenente una matrice di kadd group shap per ogni estrazione
    r-esima. Una singola matrice contiene i kadd shap per ogni features ma non 
    per le interazioni (le quali vengono calcolate ma non raccolte).
    time_ind = tempo mediamente necessario per calcolare i kadd group shap per 
    una singola istanza.
    time_set = tempo mediamente necessatio per calcolare una matrice.
    """
    instances = range(len(X))
    features = groups
    features0 = [m for n in features for m in n]
    X = np.array(X)
    #calcolo i numeri di bernoulli per il k in input
    bern = bernoulli(k)
   
    #costruisco il set di combinazioni fissando un gruppo alla volta
    all_subsets = []
    for j in range(len(features)):
        exclusive = features.copy()
        del(exclusive[j])
        exclusive = [m for n in exclusive for m in n]
        for p in range(len(exclusive) + 1):
            for element in itertools.combinations(exclusive, p):
                all_subsets.append(list(element) + features[j])
                all_subsets.append(list(element))
    
    for el in all_subsets:
        el = el.sort()
    
    sub = []
    for el in all_subsets:
        if sub.count(el) < 1:
            sub.append(el)

    #trasformo le combinazioni in vettori binari
    PM = np.zeros([len(sub), len(features0)]) 
    for l in range(len(sub)):
        PM[l, sub[l]] = 1
    
    #calcolo i kernell weights
    weights_PM = [] 
    for m in range(len(sub)):
        w = (len(features0) - 1)/(comb(len(features0), PM[m].sum())*PM[m].sum()*(len(features0) - PM[m].sum()))
        weights_PM.append(w)
    for m in range(len(weights_PM)):
        if weights_PM[m] == float('inf'):
            weights_PM[m] = 10**6
    weights_PM_norm = weights_PM/np.array(weights_PM).sum()  
    
    #definisco il massimo numero di estrazioni
    nm_max = len(sub)
    
    if nm <= nm_max:
        
        phi_array = np.ones([len(X), len(features), R])
        
        start = time.time()
        for r in range(R):
            print('simulation:', r)
            SM = np.ones([nm, len(features0)])
            SL = []
            I = range(len(sub))
            np.random.seed(r) 
            estrazioni = np.random.choice(I, size = nm, replace = False, p = weights_PM_norm) 
    
            for s in range(nm):
                SM[s] = PM[estrazioni[s]]
                sl = sub[estrazioni[s]]
                SL.append(sl)
                
            f = np.ones([len(instances), nm])
            phi0 = model.predict_proba(X_train)[:,1].mean()
        
            #calcolo il valore atteso per ogni estrazione e per ogni istanza
            for i in range(len(instances)):
                ff = []
                temp = np.tile(X[i], (len(X_train), 1)) 
                
                for l in SL:
                    mat = np.array(X_train)
                    mat[:, l] = temp[:, l]
                    val = list(model.predict_proba(mat)[:,1])
                    valS = np.array(val)
                    valS = valS.mean()
                    ff.append(valS)
        
                f[i,:] = ff
            
            f_bar = f - phi0
            
            #matrice con i kadd shap per ogni istanza e gruppo
            phi_matrix = np.ones([len(X), len(features)])

            #calcolo i kadd shap per ogni istanza e per il j-esimo gruppo
            for j in range(len(features)):
                
                #dalla lista SL delle coalzioni estratte prendo solo quelle che 
                #fanno riferimento al j-esimo gruppo
                SLJ = []
                for m in range(nm):
                    
                    if SM[m, features[j]].sum() == len(features[j]):
                        SLJ.append(SL[m])
                         
                    elif SM[m, features[j]].sum() == 0:
                        SLJ.append(SL[m])
                
                #costruisco i corrispondenti vettori binari e estraggo i valori
                #di f_bar relativi alle colaizoni scelte
                SMJ = np.zeros([len(SLJ), len(features0)]) 
                f_barJ = np.ones([len(instances), len(SLJ)])
                Z = np.ones([len(SLJ),1])
                for l in range(len(SLJ)):
                    SMJ[l, SLJ[l]] = 1
                    f_barJ[:,l] = f_bar[:, SL.index(SLJ[l])]
                    if SMJ[l, features[j]].sum() == 0:
                        Z[l] = 0
                
                #costruisco i pesi
                weights_k = np.ones(len(SLJ))
                weights_k[SLJ.index(features0)] = 10**6
                weights_k[SLJ.index([])] = 10**6
                
                ZJ = np.concatenate((SMJ[:, 0:features[j][0]], Z, SMJ[:, features[j][-1] + 1:len(features0)]), axis=1)
                W = np.diag(weights_k)
                
                #creo tutte le combinazioni con dimensione massima pari a k
                DL = []
                for d in range(k + 1):
                    for element in itertools.combinations(range(np.shape(ZJ)[1]), d):
                        DL.append(list(element))
                      
                n_par_j = len(DL)    
    
                DM = np.zeros([n_par_j, np.shape(ZJ)[1]]) 
                for ll in range(n_par_j):
                    DM[ll, DL[ll]] = 1
                
                #costruiso la trasofrmation matrix
                T = np.ones([len(SLJ), n_par_j])
                
                for m in range(len(SLJ)):
                    for n in range(n_par_j):
                        Dcard = int(sum(DM[n,:]))
                        ADcard = int(sum(DM[n,:]*ZJ[m,:]))
                        gamma = 0
                        for lll in range(ADcard + 1):
                            gamma += comb(ADcard, lll) * bern[Dcard - lll]
                        T[m,n] = gamma
                        
                S = np.linalg.inv(T.T @ W @ T) @ T.T @ W
                phi = S @ f_barJ.T
                pos = features[j][0] + 1
                phi_matrix[:, j] = phi[pos, :]
            
            phi_array[:,:,r] = phi_matrix
        
        end = time.time()
        time_cons = end - start
        #tempo medio per una singola matrice
        tempo_medio_tot = round(time_cons/R,8)
        #tempo medio per una singola instanza
        tempo_medio_ind = round(tempo_medio_tot/len(X),8)
    
        results = {'shap_array': phi_array, 'time_ind': tempo_medio_ind, 'time_set': tempo_medio_tot}
        return results
    
    else:
        print('"nm" supera il valore massimo. Valore massimo di "nm":', nm_max)

def exact_group_group_shap(X, X_train, model, groups):
    """
    Calcola l'exact group group SHAP per un vettore di istanze. Misura anche il 
    tempo medio impiegato per valutare ogni istanza.
    ----------
    PARAMETERS:
    X = desing matrix delle istanze di interesse.
    X_train  = desing matrix delle istanze usate per trainare il modello.
    model = modello da interpretare.
    groups = i gruppi di covariate di cui di vuole valutare l'impatto nelle
    predizioni.
    ------
    RETURN:
    shap_matrix = matrice con gli exact group group shap per ogni istanza e features.
    time_tot = tempo impiegato per il calcolo dell'intera matrice.
    time_ind = tempo medio impiegato per valutare la singola istanza.
    """
    instances = range(len(X))
    features = groups
    X = np.array(X)
    shap = np.ones([len(X), len(features)])
    
    start = time.time()
    
    #creo tutte le combo possibili tra i gruppi (senza trattare le variabili
    #individualmente)
    all_subsets = [] 
    for k in range(len(features), -1, -1):
        for element in itertools.combinations(range(len(features)), k):
            all_subsets.append(list(element))
    
    #trasformo la lista di combo di gruppi in una lista di combo di variabli
    #individuali
    combo = [] 
    for c in all_subsets:
        lis = [] 
        for element in c: 
            for fea in features[element]: 
                lis.append(fea) 
        combo.append(lis)
    
    #assegno un peso a ogni combinazione
    omega = []
    for l in all_subsets[1:]:
        omegal = (math.factorial(len(l)) * math.factorial(len(features) - len(l) - 1)) / math.factorial(len(features))
        omega.append(omegal)
    
    #calcolo i group gruop exact shap per ogni istanza di interesse
    for i in range(len(instances)):
        print('instance', i) 
        temp = np.tile(X[i], (len(X_train), 1))
    
        #calcolo il valore atteso relativamente all'istanza i-esima per ogni
        #combinazione
        all_val = []
        for l in combo:
            mat = np.array(X_train)
            mat[:, l] = temp[:, l]
            val = list(model.predict_proba(mat)[:,1])
            valS = np.array(val)
            valS = valS.mean()
            all_val.append(valS)
            
        #calcolo lo shap per ogni features j-esima
        for j in range(len(features)):
            Phi = 0
            for l in range(len(all_subsets)):
                if j not in all_subsets[l]:
                    valore = all_val[l]
                    peso = omega[l - 1]
                    lj = all_subsets[l] + [j]; lj.sort()
                    indice = all_subsets.index(lj)
                    valorej = all_val[indice]
                    phi = peso * (valorej - valore)
                    Phi += phi
            
            shap[i,j] = Phi
                
    end = time.time()
    #tempo per calcolare l'intera matrice
    time_tot = round(end - start, 8)
    #tempo per calcolare la singola istanza
    time_ind = round(time_tot/len(X), 8)
        
    results = {'shap_matrix': shap, 'time_tot': time_tot, 'time_ind': time_ind}
    return results

def kernell_group_group_shap(X, X_train, model, groups, nm, R):
    """
    Calcola i kernell Group Group SHAP per un vettore di istanze utilizzando 'nm' 
    coalizioni estratte dal power set di M. Ripete la procedura per R differenti 
    estrazioni di uguale dimensione 'nm'. Misura anche il tempo medio necessario 
    per fare questi calcoli.
    ----------
    PARAMETERS:
    X = desing matrix delle istanze di interesse.
    X_train  = desing matrix delle istanze usate per trainare il modello.
    model = modello da interpretare.
    groups = lista dei gruppi da analizzare.
    nm = numero di coalizioni utilizzate nel calcolo.
    R = numero di estrazioni fatte.
    ----------
    RETURN:
    shap_array = array contenente una matrice di kernell group group shap per ogni 
    estrazione r-esima. Una singola matrice contiene i kernellgroup  group shap per 
    ogni features e per ogni istanza.
    time_ind = tempo mediamente necessario per calcolare i kernell group group shap 
    per una singola istanza.
    time_set = tempo mediamente necessario per calcolare una matrice.
    """
    features = groups
    instances = range(len(X))
    X = np.array(X)
       
    #creo le possibili combinazioni tra i gruppi
    PL = []
    for k in range(len(features), -1, -1):
        for element in itertools.combinations(range(len(features)), k):
            PL.append(list(element))

    #trasformo le coalizoni in vettori binari
    PM = np.zeros([2**len(features), len(features)]) 
    for l in range(2**len(features)):
        PM[l, PL[l]] = 1

    #calcolo i pesi
    weights_PM = [] 
    for m in range(2**len(features)):
        w = (len(features) - 1)/(comb(len(features), PM[m].sum())*PM[m].sum()*(len(features) - PM[m].sum()))
        weights_PM.append(w) 
    weights_PM[0] = 10**6
    weights_PM[-1] = 10**6
    weights_PM_norm = weights_PM/np.array(weights_PM).sum()
    
    #definisco il numero massimo di estrazioni
    nm_max = len(PL)
    
    if nm <= nm_max:
        
        #array contenete le matrici di shap
        phi_array = np.ones([len(X), len(features) + 1, R])
    
        start = time.time()
        for r in range(R):
            print('simulation:', r)
            SM = np.ones([nm, len(features)])
            SL = []
            weights = np.ones([nm])
            I = range(2**len(features))
            np.random.seed(r)
            estrazioni = np.random.choice(I, size = nm, replace = False, p = weights_PM_norm) 
        
            for s in range(nm):
                SM[s] = PM[estrazioni[s]]
                weights[s] = weights_PM[estrazioni[s]]
                sl = PL[estrazioni[s]]
                SL.append(sl)
        
            Z = np.concatenate((np.ones([nm,1]), SM), axis = 1)
            W = np.diag(weights)
            S = np.linalg.inv(Z.T @ W @ Z) @ Z.T @ W
        
            #matrice contenente uno shap per ogni istanza e features
            phi_matrix = np.ones([len(X), len(features) + 1])
        
            #trasormo le coalizoni da gruppi a singole features
            combo = []
            for c in SL:
                lis = []
                for element in c:
                    for fea in features[element]:
                        lis.append(fea)
                combo.append(lis)
        
            #calcolo gli shap per l'i-esima istanza
            for i in range(len(instances)):
                f = []
                temp = np.tile(X[i], (len(X_train), 1))
            
                #calcolo gli shap per la j-esima features
                for j in combo:
                    mat = np.array(X_train)
                    mat[:, j] = temp[:, j]
                    val = list(model.predict_proba(mat)[:,1])
                    valS = np.array(val)
                    valS = valS.mean()
                    f.append(valS)
            
                phi = S @ f
                phi_matrix[i] = phi
        
            phi_array[:,:,r] = phi_matrix
    
        end = time.time()
        #tempo per tutto l'array
        time_cons = end - start
        #tempo medio per la singola matrice
        tempo_medio_tot = round(time_cons/R,8)
        #tempo medio per la singola istanza
        tempo_medio_ind = round(tempo_medio_tot/len(X),8)

        results = {'shap_array': phi_array, 'time_ind': tempo_medio_ind, 'time_set': tempo_medio_tot}
        return results
    
    else:
        print('"nm" supera il valore massimo. Valore massimo di "nm":', nm_max)

def kadd_group_group_shap(X, X_train, model, k, groups, nm, R):
    """
    Calcola i kadd group group SHAP per un vettore di istanze utilizzando 'nm' 
    coalizioni estratte dal power set di M. Ripete la procedura per R differenti
    estrazioni di uguale dimensione 'nm'. Misura anche il tempo medio necessario
    per fare questi calcoli.
    ----------
    PARAMETERS:
    X = desing matrix delle istanze di interesse.
    X_train  = desing matrix delle istanze usate per trainare il modello.
    model = modello da interpretare.
    k = grado di additività del modello.
    groups = lista dei gruppi da analizzare.
    nm = numero di coalizioni utilizzate nel calcolo.
    R = numero di estrazioni fatte.
    ----------
    RETURN:
    shap_array = array contenente una matrice di kadd group group shap per ogni 
    estrazione r-esima. Una singola matrice contiene i kadd shap per ogni 
    features e per ogni interazione fino al grado k-esimo.
    time_ind = tempo mediamente necessario per calcolare i kadd group group shap
    per una singola istanza.
    time_set = tempo mediamente necessatio per calcolare una matrice.
    """
    features = groups
    instances = range(len(X))
    X = np.array(X)
       
    PL = []
    for j in range(len(features) + 1):
        for element in itertools.combinations(range(len(features)), j):
            PL.append(list(element))
    
    DL = []
    for el in PL:
        if len(el) <= k:
            DL.append(el)
    
    n_par = len(DL)
    bern = bernoulli(k)

    PM = np.zeros([2**len(features), len(features)]) 
    for l in range(2**len(features)):
        PM[l, PL[l]] = 1
    
    DM = np.zeros([n_par, len(features)]) 
    for ll in range(n_par):
        DM[ll, DL[ll]] = 1
    
    weights_PM = [] 
    for m in range(2**len(features)):
        w = (len(features) - 1)/(comb(len(features), PM[m].sum())*PM[m].sum()*(len(features) - PM[m].sum()))
        weights_PM.append(w) 
    weights_PM[0] = 10**6
    weights_PM[-1] = 10**6
    weights_PM_norm = weights_PM/np.array(weights_PM).sum()
   
    weights_k = np.ones(2**len(features))
    weights_k[0] = 10**6
    weights_k[-1] = 10**6
    
    nm_max = len(PL)
    
    if nm <= nm_max:
        
        phi_array = np.ones([len(X), n_par, R])
    
        start = time.time()
        for r in range(R):
            print('simulation:', r)
            SM = np.ones([nm, len(features)])
            SL = []
            weights = np.ones([nm])
            I = range(2**len(features))
            np.random.seed(r)
            estrazioni = np.random.choice(I, size = nm, replace = False, p = weights_PM_norm) 
        
            for s in range(nm):
                SM[s] = PM[estrazioni[s]]
                weights[s] = weights_k[estrazioni[s]]
                sl = PL[estrazioni[s]]
                SL.append(sl)
        
            T = np.ones([nm, n_par])
        
            for m in range(nm):
                for n in range(n_par):
                    Dcard = int(sum(DM[n,:]))
                    ADcard = int(sum(DM[n,:]*SM[m,:]))
                    gamma = 0
                    for lll in range(ADcard + 1):
                        gamma += comb(ADcard, lll) * bern[Dcard - lll]
                    T[m,n] = gamma
        
            W = np.diag(weights)
            S = np.linalg.inv(T.T @ W @ T) @ T.T @ W
        
            combo = []
            for c in SL:
                lis = []
                for element in c:
                    for fea in features[element]:
                        lis.append(fea)
                combo.append(lis)
        
            phi_matrix = np.ones([len(X), n_par])
        
            for i in range(len(instances)):
                f = []
                temp = np.tile(X[i], (len(X_train), 1))
                phi0 = model.predict_proba(X_train)[:,1].mean()
            
                for j in combo:
                    mat = np.array(X_train)
                    mat[:, j] = temp[:, j]
                    val = list(model.predict_proba(mat)[:,1])
                    valS = np.array(val)
                    valS = valS.mean()
                    f.append(valS)
                
                f_bar = f - phi0
                phi = S @ f_bar
                phi_matrix[i] = phi

            phi_array[:,:,r] = phi_matrix
        
        end = time.time()
        time_cons = end - start
        tempo_medio_tot = round(time_cons/R,8)
        tempo_medio_ind = round(tempo_medio_tot/len(X),8)
    
        results = {'shap_array': phi_array, 'time_ind': tempo_medio_ind, 'time_set': tempo_medio_tot}
        return results
    
    else:
        print('"nm" supera il valore massimo. Valore massimo di "nm":', nm_max)
