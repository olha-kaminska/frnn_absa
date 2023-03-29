import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from frlearn.neighbours.classifiers import FRNN, KDTree
from frlearn.neighbours.neighbour_search_methods import NeighbourSearchMethod as NNSearch
from frlearn.uncategorised.weights import Weights as OWAOperator
from frlearn.uncategorised.weights import LinearWeights as additive
from frlearn.base import probabilities_from_scores
from frlearn.classifiers import FRNN
from frlearn.feature_preprocessors import VectorSizeNormaliser
from frlearn.vector_size_measures import MinkowskiSize
from frlearn.base import select_class
from frlearn.classifiers import FROVOCO
from frlearn.weights import ReciprocallyLinearWeights


def frnn_owa_method(train_data, y, test_data, vector_name, NNeighbours, lower, upper):
    '''
    This function implements FRNN OWA classification method 
    
    This function uses numpy package as np.
    
    Input:  train_data - train data in form of pandas DataFrame 
            y - the list of train_data golden labels 
            test_data - train data in form of pandas DataFrame 
            vector_name - string, the name of vector with features in train_data and test_data
            NNeighbours - the int number of neighbours for FRNN OWA method
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
                            possible options: strict(), exponential(), invadd(), mean(), additive()
    Output: conf_scores - the list of confidence scores, y_pred - the list of predicted labels
    '''
    train_ind = list(train_data.index)
    test_ind = list(test_data.index)
    X = np.zeros((len(train_data), len(train_data[vector_name][train_ind[0]])))
    for m in range(len(train_ind)):
        for j in range(len(train_data[vector_name][train_ind[m]])):
            X[m][j] = train_data[vector_name][train_ind[m]][j]
    X_test = np.zeros((len(test_data), len(test_data[vector_name][test_ind[0]])))
    for k in range(len(test_data)):
        for j in range(len(test_data[vector_name][test_ind[0]])):
            X_test[k][j] = test_data[vector_name][test_ind[k]][j]
    OWA = OWAOperator()(NNeighbours)
    
    clf = FRNN(
    dissimilarity=MinkowskiSize(p=2, unrooted=True),
    preprocessors=(VectorSizeNormaliser('euclidean'), ),
        lower_weights=lower, lower_k=NNeighbours, upper_k=NNeighbours)  
    cl = clf.construct(X, y)     
    # confidence scores
    conf_scores = cl.query(X_test)
    # labels 
    y_pred = np.argmax(conf_scores, axis=1)
    return conf_scores, y_pred

def test_ensemble_labels(train_data, y, test_data, vector_names, NNeighbours, lower, upper):
    '''
    This function performs ensemble of FRNN OWA methods based on labels as output
    
    This function uses numpy package as np.
    
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances 
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            vector_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that will be used to perform    
                           FRNN OWA classification method for the corresponded feature vector. Lenghts of 'vector_names' and 'NNeighbours' lists should be equal. 
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
                            possible options: strict(), exponential(), invadd(), mean(), additive()
    Output: y_pred_res - the list of predicted labels
    '''
    y_pred = []
    for j in range(len(vector_names)):       
        y_pred.append(frnn_owa_method(train_data, y, test_data, vector_names[j], NNeighbours[j], lower, upper)[1])
    # Use voting function to obtain the ensembled label - we used mean 
    y_pred_res = np.mean(y_pred, axis=0)
    return y_pred_res
    
def weights_sum_test(conf_scores, class_num, alpha=0.8):    
    '''
    This function performs rescaling and softmax transformation of confidence scores 
    
    This function uses numpy package as np.
    
    Input:  conf_scores - the list of confidence scores
            class_num - the integer number of classes
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: the list of transformed confidence scores
    '''
    conf_scores_T = conf_scores.T
    conf_scores_T_rescale = [[(conf_scores_T[k][i]-0.5)/(alpha) for i in range(len(conf_scores_T[k]))] for k in range(class_num)]
    conf_scores_T_rescale_sum = [sum(conf_scores_T_rescale[k]) for k in range(class_num)]
    res = [np.exp(conf_scores_T_rescale_sum[k])/sum([np.exp(conf_scores_T_rescale_sum[k]) for k in range(class_num)]) for k in range(class_num)]
    return res
    
def test_ensemble_confscores(train_data, y, test_data, vector_names, class_name, NNeighbours, lower, upper, alpha=0.8):
    '''
    This function performs ensemble of FRNN OWA methods based on confidence scores outputs
    
    This function uses numpy package as np.
    
    Input:  train_data - pandas DataFrame with train data that contains features 'vector_names'
            y - pandas Series or list, golden labels of train_data instances 
            test_data - pandas DataFrame with test data that contains features 'vector_names'
            vector_names - the list of strings, names of features vectors in train_data and test_data
            NNeighbours - the list of int numbers, where each number represent amount of neighbours 'k' that will be used to perform    
                           FRNN OWA classification method for the corresponded feature vector. Lenghts of 'vector_names' and 'NNeighbours' lists should be equal. 
            lower, upper - lower and upper approximations, calculated with Ordered Weighted Average (OWA) aggregation operators
                            possible options: strict(), exponential(), invadd(), mean(), additive()
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: y_pred_res - the list of predicted labels
    '''
    # Calculate number of classes
    class_num = len(set(train_data[class_name]))
    # Create and fill 3D array
    conf_scores_all = np.zeros((len(vector_names), len(test_data), class_num))
    for j in range(len(vector_names)):    
        # Calculate confidence scores for each feature vector 
        result = frnn_owa_method(train_data, y, test_data, vector_names[j], NNeighbours[j], lower, upper)[0]
        # Check for NaNs 
        for k in range(len(result)):
            if np.any(np.isnan(result[k])):
                result[k] = [0 for i in range(class_num)]              
        conf_scores_all[j] = (result)
    # Rescale obtained confidence scores 
    rescaled_conf_scores = np.array([weights_sum_test(conf_scores_all[:, k, :], class_num, alpha) for k in range(len(conf_scores_all[0]))])
    # Use the mean voting function to obtain the predicted label 
    y_pred_res = [np.round(np.average(k, weights=list(set(train_data[class_name])))) for k in rescaled_conf_scores]
    return y_pred_res
    
def cross_validation_ensemble_owa(df, features_names, class_name, K_fold, k, lower, upper, method, evaluation, alpha=0.8):
    '''
    This function performs cross-validation evaluation for FRNN OWA ensemble.
    
    It uses numpy library as np for random permutation of list.
    
    Input:  df - pandas DataFrame with features to evaluate 
            features_names - the list of strings, names of features vectors in df
            class_name - the string name of the column of df that contains classes of instances 
            K_fold - the number of folds of cross-validation, we used K_fold = 5
            k - the list of int numbers, where each number represent amount of neighbours 'k' that will be used 
                to perform FRNN OWA classification method for the corresponded feature vector. Lenghts of 'features_names' and 'k' lists should be equal
            method - this string variable defines the output of wkNN approach, it can be 'labels' or 'conf_scores'
            evaluation - the evaluation method's name: could be 'pcc' for Pearson Correlation Coefficient or 'f1' for F1-score
            alpha - the int parameter used for confidence scores rescaling, by default it is equal to 0.8 
    Output: The evaluation score as float number: either PCC or F1-score             
    '''
    # Create column for results
    df[method] = None

    # Cross-validation
    random_indices = np.random.permutation(df.index)
    for i in range(K_fold): 
        # Split df on train and test data
        test_data = df.loc[df.index.isin(random_indices[i*len(df.index)//K_fold:(i+1)*len(df.index)//K_fold])]
        train_data = df[~df.index.isin(test_data.index)]
        y = train_data[class_name]
        y_true = test_data[class_name]       
        # Apply FRNN OWA method for each feature vector depends on specified output type 
        if method == 'labels':
            # Solution for labels calculation 
            y_pred_res = test_ensemble_labels(train_data, y, test_data, features_names, k, lower, upper)
        elif method == 'conf_scores':
            # Solution for confidence scores calculation 
            y_pred_res = test_ensemble_confscores(train_data, y, test_data, features_names, class_name, k, lower, upper, alpha)
        else:
            print('Wrong output type was specified!')
        df[method].loc[test_data.index] = y_pred_res
    if evaluation == 'macro':
        p, r, f1, support = precision_recall_fscore_support(df[class_name].to_list(), df[method].to_list(), average = "macro")
    if evaluation == 'weighted':
        p, r, f1, support = precision_recall_fscore_support(df[class_name].to_list(), df[method].to_list(), average = "weighted")
        return f1
        
def frovoco_method(train_data, y, test_data, vector_name, nnk):
    '''
    This function performs cross-validation evaluation for FROVOCO method.
    
    It uses numpy library as np.
    
    Input:  train_data - pandas DataFrame with train data 
            y - list of real classes 
            test_data - pandas DataFrame with test data 
            vector_name - name of features vectors in train_data/test_data 
            nnk - integer that represents amount of neighbours 'k' 
    Output: y_pred - list of predictions             
    '''
    train_ind = list(train_data.index)
    test_ind = list(test_data.index)
    X = np.zeros((len(train_data), len(train_data[vector_name][train_ind[0]])))
    for m in range(len(train_ind)):
        for j in range(len(train_data[vector_name][train_ind[m]])):
            X[m][j] = train_data[vector_name][train_ind[m]][j]
    X_test = np.zeros((len(test_data), len(test_data[vector_name][test_ind[0]])))
    for k in range(len(test_data)):
        for j in range(len(test_data[vector_name][test_ind[0]])):
            X_test[k][j] = test_data[vector_name][test_ind[k]][j]
    
    clf = FROVOCO(
    imbalanced_weights=LinearWeights(), # linear = additive weights
    imbalanced_k=nnk,#multiple(0.1), # you can enter a number like 20, but multiple(0.1) means the 10% closest instances
    balanced_weights=LinearWeights(),#ReciprocallyLinearWeights(), # = inverse additive weights
    balanced_k=nnk,#log_multiple(3), # means 3 times the log of the number of instances
    ir_threshold=10, # the imbalance ratio above which to use imbalanced weights. Default is 9 from the paper. Use None to only use balanced weights
    dissimilarity=MinkowskiSize(p=2, unrooted=True),
    preprocessors=(VectorSizeNormaliser('euclidean'), ),
    )
    model = clf(X, y)
    y_pred = model(X_test)
    return y_pred

def test_ensemble_frovoco(train_data, y, test_data, vector_names, nnk_values):
    '''
    This function performs test for FROVOCO method.
    
    This function uses numpy package as np
    
    Input:  train_data - pandas DataFrame with train data 
            y - list of real classes 
            test_data - pandas DataFrame with test data 
            vector_names - a list of names of features vectors in train_data/test_data 
            nnk_values - a list of integers that represents amount of neighbours 'k' 
    Output: y_pred_res - list of predictions             
    '''
    y_pred = []
    for j in range(len(vector_names)):       
        y_pred.append(frovoco_method(train_data, y, test_data, vector_names[j], nnk_values[j]))
    # Use voting function to obtain the ensembled label - we used mean 
    y_pred_res = np.mean(y_pred, axis=0)
    return np.argmax(y_pred_res, axis=1)

def cross_validation_ensemble_frovoco(df, vector_names, class_name, K_fold, nnk_values):
    '''
    This function performs cross-validation evaluation for FROVOCO ensemble.
    
    This function uses numpy package as np
    
    Input:  train_data - pandas DataFrame with train data 
            y - list of real classes 
            test_data - pandas DataFrame with test data 
            vector_names - a list of names of features vectors in train_data/test_data 
            nnk_values - a list of integers that represents amount of neighbours 'k' 
    Output: y_pred_res - list of predictions             
    '''
    f1_list = []
    random_indices = np.random.permutation(df.index)
    for i in range(K_fold): 
        # Split df on train and test data
        test_data = df.loc[df.index.isin(random_indices[i*len(df.index)//K_fold:(i+1)*len(df.index)//K_fold])]
        train_data = df[~df.index.isin(test_data.index)]
        y = train_data[class_name]
        y_true = test_data[class_name]       
    
        # Solution for labels calculation 
        y_pred_res = test_ensemble_frovoco(train_data, y, test_data, vector_names, nnk_values)
        y_pred_res = np.argmax(y_pred_res, axis=1)
        # Calculate F1
        #p_irony, r_irony, f1_irony, support = precision_recall_fscore_support(y_true, y_pred_res, average = "macro")
        p_irony, r_irony, f1_irony, support = precision_recall_fscore_support(y_true, y_pred_res, average = "weighted")
        f1_list.append(f1_irony)
    return np.mean(f1_list)
    
def calculate_cost_corrected_accuracy(labels, conf_matrix, cost_matrix_fp):
    """
    This function calculates the cost corrected accuracy
    
    It uses json package and numpy as np. 
    
    Input: - labels: the list of the unique original labels
           - conf_matrix: the np array of the confusion matrix 
           - cost_matrix_fp: string, the path to the json containing the cost for each label combination
    Output: cc_acc: float score 
    """
    with open(cost_matrix_fp, 'r', encoding='utf-8') as cost_json:
        cost_dict = json.load(cost_json)
    cost_rows = []
    for label in sorted(labels):
        cost_row = [v for k, v in cost_dict[label].items() if k in labels]
        cost_rows.append(cost_row)
    conf_m = np.array(conf_matrix)
    cost_m = np.array(cost_rows)
    cost = np.sum(np.multiply(conf_m, cost_m) / np.sum(conf_m))
    cc_acc = 1 - cost
    return cc_acc

def cosine_relation(tweet1, tweet2):
    '''
    This function calculates cosine relation
    
    Input: tweet1, tweet2 - arrays of numbers
    output: float, cosine relation
    '''
    return 0.5 * (1 + np.dot(tweet1, tweet2)/(np.linalg.norm(tweet1)*np.linalg.norm(tweet2)))
    
def get_neigbours(test_vector, df_train_vectors, feature, k, text_column, class_column): 
    '''
    This function calculates k neirest neighbours to the test_vector using cosine similarity
    
    Input: test_vector - array of numbers
           df_train_vectors - DataFrame with train instances
           feature - name of a column in df_train_vectors with texts' embedding vectors
           k - a number of neirest neighbours 
           text_column - name of a column in df_train_vectors with texts
           class_column - name of a column in df_train_vectors with texts' classes
    Output: list of k neirest neighbours' texts and list with their classes
    '''
    distances = df_train_vectors[feature].apply(lambda x: cosine_relation(x, test_vector))
    top_k = distances.sort_values(ascending=False)[:k]
    df_top_k = df_train_vectors.loc[top_k.index]
    return df_top_k[text_column].to_list(), df_top_k[class_column].to_list()