import json

def system_0(train, test, [class_aspect, class_sen, class_emo], [vector_name_asp, vector_name_sen, vector_name_emo], [k_asp, k_sen, k_emo]):
    '''
    This function performs basic system pipeline, where aspect, sentiment, and emotion tasks are performed one after one.
    
    Input: - train: dataframe with train dataset
           - test: dataframe with test dataset
           - [class_aspect, class_sen, class_emo]: list of strings, which represent name of columns in train (test) dataset with classes that we will use for aspect/sentiment/emotion tasks 
           - [vector_name_asp, vector_name_sen, vector_name_emo]: list of strings, which represent name of columns in train (test) dataset with feature vectors that we will use for aspect/sentiment/emotion tasks 
           - [k_asp, k_sen, k_emo]: list of integers, which represents parameter k (a number of neighbours) that we will use for aspect/sentiment/emotion tasks 
           
    Output: three lists with predictions for aspect/sentiment/emotion tasks
    '''
    #aspect classification
    res_asp = test_ensemble_labels(train, train[class_aspect], test, [vector_name_asp], [k_asp], additive(), additive())
    #filter out wrong predictions 
    correct_asp = []
    for i in range(len(test)):
        if test[class_aspect][i] == res_asp[i]:
            correct_asp.append(i)
    test1 = test[test.index.isin(correct_asp)]
    #sentiment
    res_sen = test_ensemble_frovoco(train, train[class_sen], test1, [vector_name_sen], [k_sen])
    #filter 
    correct_sen = []
    indxs = test1.index.to_list()
    for i in range(len(test1)):
        if test1['Class_sentiment'][indxs[i]] == res_sen[i]:
            correct_sen.append(indxs[i])
    test2  = test1[test1.index.isin(correct_sen)]
    #emotion
    res_emo = test_ensemble_frovoco(train, train[class_emo], test2, [vector_name_emo], [k_emo])
    #fill the gaps in the original test dataset
    test_copy = test.copy()
    test_copy['Predicted_emo_label'] = None
    test_copy['Predicted_sen_label'] = None
    for i in test_copy.index.to_list():
        if i in test1.index.to_list():
            test_copy['Predicted_sen_label'][i] = test1['Predicted_sen_label'][i]
        else:
            test_copy['Predicted_sen_label'][i] = 100
        if i in test2.index.to_list():
            test_copy['Predicted_emo_label'][i] = test2['Predicted_emo_label'][i]
        else: 
            test_copy['Predicted_emo_label'][i] = 100
    
    return res_asp, list(test_copy['Predicted_sen_label']), list(test_copy['Predicted_emo_label'])
    
def system_1(train, test, [class_aspect, class_sen, class_emo], [vector_name_asp, vector_name_sen, vector_name_emo_pos, vector_name_emo_neg], [k_asp, k_sen, k_emo_pos, k_emo_neg]):
    '''
    This function performs system #1, which is an updated version of system #0, where as class_aspect we should use main aspect classes and for emotions we created two modes: one for positive emotions and one for negative 
    
    Input: - train: dataframe with train dataset
           - test: dataframe with test dataset
           - [class_aspect, class_sen, class_emo]: list of strings, which represent name of columns in train (test) dataset with classes that we will use for aspect/sentiment/emotion tasks 
           - [vector_name_asp, vector_name_sen, vector_name_emo_pos, vector_name_emo_neg]: list of strings, which represent name of columns in train (test) dataset with feature vectors that we will use for aspect/sentiment/positive emotion/negative emotion tasks 
           - [k_asp, k_sen, k_emo_pos, k_emo_neg]: list of integers, which represents parameter k (a number of neighbours) that we will use for aspect/sentiment/positive emotion/negative emotion tasks 
           
    Output: three lists with predictions for aspect/sentiment/emotion tasks
    '''
    #aspect classification
    res_asp = test_ensemble_labels(train, train[class_aspect], test, [vector_name_asp], [k_asp], additive(), additive())
    #filter out wrong predictions 
    correct_asp = []
    for i in range(len(test)):
        if test[class_aspect][i] == res_asp[i]:
            correct_asp.append(i)
    test1 = test[test.index.isin(correct_asp)]
    #sentiment
    res_sen = test_ensemble_frovoco(train, train[class_sen], test1, [vector_name_sen], [k_sen])
    #filter 
    correct_sen = []
    indxs = test1.index.to_list()
    for i in range(len(test1)):
        if test1['Class_sentiment'][indxs[i]] == res_sen[i]:
            correct_sen.append(indxs[i])
    test2  = test1[test1.index.isin(correct_sen)]
    #emotion: 2 models
    test2_pos = test2.loc[test2['sentiment'].isin(['very_pos', 'pos'])]
    test2_neg = test2.loc[test2['sentiment'].isin(['very_neg', 'neg'])]
    test2_neu = test2.loc[test2['sentiment'].isin(['neu'])]
    #positive prediction 
    res_pos = test_ensemble_labels(train, train[class_emo], test2_pos, [vector_name_emo_pos], [k_emo_pos], additive(), additive())
    #negative predictions
    res_neg = test_ensemble_frovoco(train, train[class_emo], test2_neg, [vector_name_emo_neg], [k_emo_neg])
    #fill the gaps for the original test dataset
    test2_pos['Predicted_emo_label'] = res_pos
    test2_neg['Predicted_emo_label'] = res_neg
    test1['Predicted_sen_label'] = res_sen
    test_copy = test.copy()
    test_copy['Predicted_emo_label'] = None
    test_copy['Predicted_sen_label'] = None
    for i in test_copy.index.to_list():
        if i in test1.index.to_list():
            test_copy['Predicted_sen_label'][i] = test1['Predicted_sen_label'][i]
        else:
            test_copy['Predicted_sen_label'][i] = 100
        if i in test2_pos.index.to_list():
            test_copy['Predicted_emo_label'][i] = test2_pos['Predicted_emo_label'][i]
        elif i in test2_neg.index.to_list():
            test_copy['Predicted_emo_label'][i] = test2_neg['Predicted_emo_label'][i]
        elif i in test2_neu.index.to_list():
            test_copy['Predicted_emo_label'][i] = 7
        else: 
            test_copy['Predicted_emo_label'][i] = 100
        
    return res_asp, list(test_copy['Predicted_sen_label']), list(test_copy['Predicted_emo_label'])
    
def system_2(train, test, [class_aspect, class_sen, class_emo], [vector_name_asp, vector_name_sen, vector_name_emo_pos, vector_name_emo_neg], [k_asp, k_sen, k_emo_pos, k_emo_neg], ct_pol_path):
    '''
    This function performs system #2, which is an updated version of system #1, where for results of sentiment task we perform filtration with a usage of the cost scores. 
    
    This function uses json library.
    
    Input: - train: dataframe with train dataset
           - test: dataframe with test dataset
           - [class_aspect, class_sen, class_emo]: list of strings, which represent name of columns in train (test) dataset with classes that we will use for aspect/sentiment/emotion tasks 
           - [vector_name_asp, vector_name_sen, vector_name_emo_pos, vector_name_emo_neg]: list of strings, which represent name of columns in train (test) dataset with feature vectors that we will use for aspect/sentiment/positive emotion/negative emotion tasks 
           - [k_asp, k_sen, k_emo_pos, k_emo_neg]: list of integers, which represents parameter k (a number of neighbours) that we will use for aspect/sentiment/positive emotion/negative emotion tasks 
           - ct_pol_path: string, path to the cost matrix for sentiment classes 
           
    Output: three lists with predictions for aspect/sentiment/emotion tasks
    '''
    #aspect classification
    res_asp = test_ensemble_labels(train, train[class_aspect], test, [vector_name_asp], [k_asp], additive(), additive())
    #filter out wrong predictions 
    correct_asp = []
    for i in range(len(test)):
        if test[class_aspect][i] == res_asp[i]:
            correct_asp.append(i)
    test1 = test[test.index.isin(correct_asp)]
    #sentiment
    res_sen = test_ensemble_frovoco(train, train[class_sen], test1, [vector_name_sen], [k_sen])
    #filter 
    correct_sen = []
    indxs = test1.index.to_list()
    with open(ct_pol_path, 'r', encoding='utf-8') as cost_json:
        cost_dict = json.load(cost_json)
    for i in range(len(test1)):
        cost_one = cost_dict[res_sen_MC_pol[i]][test1['sentiment'][indxs[i]]]
        if cost_one<1:
            correct_sen.append(indxs[i])   
    test2 = test1[test1.index.isin(correct_sen)]
    #emotion: 2 models
    test2_pos = test2.loc[test2['sentiment'].isin(['very_pos', 'pos'])]
    test2_neg = test2.loc[test2['sentiment'].isin(['very_neg', 'neg'])]
    test2_neu = test2.loc[test2['sentiment'].isin(['neu'])]
    #positive prediction 
    res_pos = test_ensemble_labels(train, train[class_emo], test2_pos, [vector_name_emo_pos], [k_emo_pos], additive(), additive())
    #negative predictions
    res_neg = test_ensemble_frovoco(train, train[class_emo], test2_neg, [vector_name_emo_neg], [k_emo_neg])
    #fill the gaps for the original test dataset
    test2_pos['Predicted_emo_label'] = res_pos
    test2_neg['Predicted_emo_label'] = res_neg
    test1['Predicted_sen_label'] = res_sen
    test_copy = test.copy()
    test_copy['Predicted_emo_label'] = None
    test_copy['Predicted_sen_label'] = None
    for i in test_copy.index.to_list():
        if i in test1.index.to_list():
            test_copy['Predicted_sen_label'][i] = test1['Predicted_sen_label'][i]
        else:
            test_copy['Predicted_sen_label'][i] = 100
        if i in test2_pos.index.to_list():
            test_copy['Predicted_emo_label'][i] = test2_pos['Predicted_emo_label'][i]
        elif i in test2_neg.index.to_list():
            test_copy['Predicted_emo_label'][i] = test2_neg['Predicted_emo_label'][i]
        elif i in test2_neu.index.to_list():
            test_copy['Predicted_emo_label'][i] = 7
        else: 
            test_copy['Predicted_emo_label'][i] = 100
        
    return res_asp, list(test_copy['Predicted_sen_label']), list(test_copy['Predicted_emo_label'])
    
def system_3(train, test, [class_aspect, class_sen, class_emo], [vector_name_asp, vector_name_sen, vector_name_emo_pos, vector_name_emo_neg], [k_asp, k_sen, k_emo_pos, k_emo_neg]):
    '''
    This function performs system #3, which is an updated version of system #1, where for emotions we created two modes: one for positive emotions and one for negative, but all tasks (aspect, sentiment, emotion) are performed separately 
    
    Input: - train: dataframe with train dataset
           - test: dataframe with test dataset
           - [class_aspect, class_sen, class_emo]: list of strings, which represent name of columns in train (test) dataset with classes that we will use for aspect/sentiment/emotion tasks 
           - [vector_name_asp, vector_name_sen, vector_name_emo_pos, vector_name_emo_neg]: list of strings, which represent name of columns in train (test) dataset with feature vectors that we will use for aspect/sentiment/positive emotion/negative emotion tasks 
           - [k_asp, k_sen, k_emo_pos, k_emo_neg]: list of integers, which represents parameter k (a number of neighbours) that we will use for aspect/sentiment/positive emotion/negative emotion tasks 
           
    Output: three lists with predictions for aspect/sentiment/emotion tasks
    '''
    #aspect classification
    res_asp = test_ensemble_labels(train, train[class_aspect], test, [vector_name_asp], [k_asp], additive(), additive())
    #sentiment
    res_sen = test_ensemble_frovoco(train, train[class_sen], test, [vector_name_sen], [k_sen])
    #emotion: 2 models
    test_pos = test.loc[test['sentiment'].isin(['very_pos', 'pos'])]
    test_neg = test.loc[test['sentiment'].isin(['very_neg', 'neg'])]
    test_neu = test.loc[test['sentiment'].isin(['neu'])]
    #positive prediction 
    res_pos = test_ensemble_labels(train, train[class_emo], test2_pos, [vector_name_emo_pos], [k_emo_pos], additive(), additive())
    #negative predictions
    res_neg = test_ensemble_frovoco(train, train[class_emo], test2_neg, [vector_name_emo_neg], [k_emo_neg])
    #fill the gaps for the original test dataset
    test_copy = test.copy()
    test_copy['Predicted_emo_label_sys3'] = None

    test_MC_neg['Predicted_emo_labels'] = np.argmax(res_neg_MC, axis=1)
    test_MC_pos['Predicted_emo_labels'] = res_pos_MC
    test_copy['Predicted_emo_label'] = None
    for i in test_copy.index.to_list():
        if i in test_MC_pos.index.to_list():
            test_copy['Predicted_emo_label'][i] = test_MC_pos['Predicted_emo_labels'][i]
        elif i in test_MC_neg.index.to_list():
            test_copy['Predicted_emo_label'][i] = test_MC_neg['Predicted_emo_labels'][i]
        elif i in test_MC_neu.index.to_list():
            test_copy['Predicted_emo_label'][i] = 7
        else: 
    return res_asp, res_sen, list(test_copy['Predicted_emo_label'])