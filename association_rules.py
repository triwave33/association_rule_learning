# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product, combinations
from numba import jit, njit


class Rule:
    def __init__(self, ant, con):
        self.ant= ant
        self.con = con
        
    def print(self):
        print("Rule(%s, %s)" % (str(self.ant), str(self.con)))

## Define functions

# return boolean 
def GTI(df, indexes): # Get True Indexes
    '''
    Input
        df: dataframe (row: transactions, col: items)
        index: column indexes num to be examined 
    Output
        list of boolen
    '''
    if type(indexes) == int:
        indexes = [indexes]
    elif type(indexes) == np.int64:
        indexes = [indexes]
        
    return df[:,indexes]==1


def support(array_2d,  indexes, m='num'):
    gti_b = GTI(array_2d, indexes)
    if gti_b.shape[1]==0:
        return 0
    b = np.all(gti_b, axis=1)
    if m =='num':
        return np.sum(b)
    elif m =='ratio':
        return np.sum(b)/len(b)
    elif m == 'bool':
        return b


def confidence(array_2d, X_indexes, Y_indexes):
    sup_X = support(array_2d, X_indexes)
    X_Y_indexes = np.hstack([X_indexes, Y_indexes])
    return support(array_2d, X_Y_indexes)/sup_X
    

def getF1(array_2D, minsup):
    '''
    Description
        Output frequent itemset from original tables (k=1)
    Input
        array_2D -> row: transactions, col: items
    Output
        frequent items' indexes: list
    '''
    return np.array([[col] for col in range(array_2D.shape[1]) if support(array_2D,col, m='ratio') >= minsup])


def getFkPlusOne(array_2D, indexes, minsup):
    '''
    Description
        Output frequent itemset from original tables (k=1)
    Input
        array_2D -> row: transactions, col: items
        indexes -> list of tuples [(),(),...()] 
    '''
    
    return np.array([col for col in indexes if support(array_2D,col, m='ratio') >= minsup])


def getCkPlusOne(prevCandidate, k):
    '''
    input
    prevCandidate: [(),(),...,()] # list of tuples
    k: length of next Candidate
    Output
      nextCandidate:[(),(),...,()] # list of tuples
     '''
    
    assert np.all(np.array([len(x) for x in prevCandidate])== k-1)
    assert k >1
    items = list(np.unique(np.array(prevCandidate).flatten())) 
    tmp_candidates = [x for x in combinations(items, k)]
    if k ==2:
        return np.array(tmp_candidates)
        
    candidates = [
        candidate for candidate in tmp_candidates
        if all(
            x in prevCandidate
            for x in combinations(candidate, k - 1))
    ]
       
    return np.array(candidates)


def isEmpty(F):
    if len(F) < 1:
        return True
    else:
        return False


def isCalcConfNeeded(array_prev_ant, array_ant, set_f):
    
        '''
        Input 
            array_prev_ant: array([1,2], [1,3], [2,3]) <- example
            array_ant: array([1],[2],[3])
            set_f: [1,2,3]
            
        Var
            array_prev_con: array([3],[2],[1])
            array_con: array([2,3,[1,3],[1,2])
        
        '''
        array_prev_con  = np.array([tuple(set_f - set(c))  for c in array_prev_ant])
        array_con  = np.array([tuple(set_f - set(c))  for c in array_ant])
        
        out = []
        
        for a,c in zip(array_ant, array_con):
            out_inner = []
            for i in range(len(c)):
                #array_ant_candidate = a
                cand = c[i]
                array_candidate_ant = np.append(a, cand)
                array_candidate_con = np.delete(c, i)
                
                res = np.any([set(array_candidate_ant) <= set(i)  for i in array_prev_ant])
                out_inner.append(res)
            if np.all(out_inner):
                out.append(True)
            else:
                out.append(False)
            
        return out


def frequent(db, minsum):
    
    k = 1
    F_list = []
    F_now = getF1(db, minsum)
    F_list.append(F_now)
    while(True):

        C_next = getCkPlusOne(F_now, k+1)

        F_next = getFkPlusOne(db, C_next, minsum)
        if isEmpty(F_next):
            break
        k += 1
        F_now = F_next
        F_list.append(F_now)
        
    return F_list


def find_rules(db, F_list, minconf) :


    conf_list = []
    for F in F_list: # F_listの中の特定のkに対するfrequent items
        k = len(F[0])
        if k == 1:
            conf_list.append([None])
            pass
        elif k == 2:
            conf_list_k = []
            for f_2 in F:
                A = f_2[0]
                B = f_2[1]
                conf_AB = confidence(db, A, B)
                if conf_AB >= minconf:
                    #conf_list_k.append((np.array([A]),np.array([B])))
                    conf_list_k.append(Rule([A],[B]))
                conf_BA = confidence(db, B, A)
                if conf_BA >= minconf:
                    #conf_list_k.append((np.array([B]),np.array([A])))
                    conf_list_k.append(Rule([B],[A]))

            conf_list.append(conf_list_k)
        
        elif k >= 3:
            conf_list_k = []
            for f_k in F:    # 特定のkに対する一つ一つのfrequent item
               
                
                '''example
                k=3
                
                j=1
                f_k: [1,2,3]
                conf: ([1,2],[3]), ([1,3],[2]),([2,3],[1]) # j = 1
                f_kからj=1個consequentを作りだしてconfidenceを求める
                confidenceがminconf以上のものをconf_listに格納
                格納するものを
                ([1,2],[3]),([2,3],[1])とする(※とする)。　([1,3],[2])はminconf未満
                
                j = 2
                f_k: [1,2,3]
                conf: ([1],[2,3]),([2],[1,3]),([3],[1,2])
                リストアップしたconfの全てに対してconfidenceを計算すると時間がかかるので、
                効率的に解く。
                
                例えば([1],[2,3])のconfidenceを計算する必要があるのか？については
                consequentから一つをantecedentに戻したもの全てが※に含まれる必要がある
                
                [1],[2,3]からは
                [1,2],[3]　および [1,3],[2]が得られる。conf_listに格納した値は
                [1,2],[3] および　[2,3],[1]　であるので条件を満たさない
                故に[1],[2,3]はconfidenceを計算する必要がない。
                
                [2],[1,3]からは
                [1,2],[3]　および [2,3],[1]が得られる。conf_listに格納した値は
                [1,2],[3] および　[2,3],[1]　であるので条件を満たす
                故に[2],[1,3]はconfidenceを上回る可能性があるので計算する必要がある。
                
                上記の探索は結局 対象としている相関ルールのantecedentが前回の※の全ての
                antecedentに含まれていることに帰着できる。
                
                '''
                
                set_all = set(f_k)
                j= 1 # k個のantecedentからj個をconsequentに移動させてconfidenceを算出する
                
                # 新しい相関ルール候補の作成
                array_antecedent = np.array(list(combinations(f_k, k-1 )))
                array_consequent = np.array([tuple(set_all - set(c))  for c in array_antecedent])
                conf = np.array([confidence(db, ant, con) for ant, con in zip(array_antecedent, array_consequent)])
                
                isHigher = conf >= minconf
                if sum(isHigher) > 0:
                    array_antecedent_filtered_by_conf = array_antecedent[isHigher]
                    array_consequent_filtered_by_conf = array_consequent[isHigher]
                    
                    conf_list_k.append([Rule(a,c) for a,c in zip(array_antecedent_filtered_by_conf, array_consequent_filtered_by_conf)])
             
               
               
                    while(j < k-1):
                        array_antecedent_new = np.array(list(combinations(f_k, k-(j+1) )))
                        # filter antecedent by previous conf
                        
                        _res = isCalcConfNeeded(array_antecedent_filtered_by_conf, array_antecedent_new, set_all)
                        
                        array_antecedent_filtered_by_prev = array_antecedent_new[_res]
                        array_consequent_filtered_by_prev = np.array([tuple(set_all - set(c))  for c in array_antecedent_filtered_by_prev])
                        conf = np.array([confidence(db, ant, con) for ant, con in zip(array_antecedent_filtered_by_prev, 
                                                                                                                                  array_consequent_filtered_by_prev)])
                        isHigher = conf >= minconf
                        if sum(isHigher) > 0:
                            array_antecedent_filtered_by_prev_and_conf = array_antecedent_filtered_by_prev[isHigher]
                            array_consequent_filtered_by_prev_and_conf = array_consequent_filtered_by_prev[isHigher]
                            
                            conf_list_k.append([Rule(a,c) for a,c in zip(array_antecedent_filtered_by_prev_and_conf, 
                                                                                                 array_consequent_filtered_by_prev_and_conf)])
                        
                        j += 1
            
            conf_list.append(conf_list_k)
    
    return conf_list
                



