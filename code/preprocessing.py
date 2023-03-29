import torch
import tensorflow as tf

def get_term(token, indexes_str):
    '''
    This function extract the term out of the tokens.
    
    Input: - token: list of words (strings)
           - indexes_str: list of int, that identify the position of targeted term in the token list (these numbers should be <= the lenght o token list)
    Output: - term: string 
    '''
    words = token.replace('[', '').replace(']', '').replace('"\'",', '@').replace("\'", '').replace(",", '').replace('\"', '').replace('@', '\'').split(' ')
    indexes_list = [int(i) for i in indexes_str.replace('[', '').replace(']', '').replace(' ', '').split(',')]
    term = " ".join([words[i] for i in indexes_list])
    return term
    
def window_3(tokens, indexes):
    '''
    This function extract the window of tokens around the term size 3.
    
    Input: - tokens: list of words (strings)
           - indexes: list of int, that identify the position of targeted term in the token list (these numbers should be <= the lenght o token list)
    Output: - window3: list of strings
    '''
    window3 = []
    if indexes[0]-3>0:\
        start = indexes[0]-3
    else:
        start = 0 
    if indexes[-1]+4<len(tokens):
        end = indexes[-1]+4
    else:
        end = len(tokens)
    for i in range(start, end):
        window3.append(tokens[i])
    return window3
        
def window_5(tokens, indexes):
    '''
    This function extract the window of tokens around the term size 5.
    
    Input: - tokens: list of words (strings)
           - indexes: list of int, that identify the position of targeted term in the token list (these numbers should be <= the lenght o token list)
    Output: - window5: list of strings
    '''
    window5 = []
    if indexes[0]-5>0:
        start = indexes[0]-5
    else:
        start = 0 
    if indexes[-1]+6<len(tokens):
        end = indexes[-1]+6
    else:
        end = len(tokens)
    for i in range(start, end):
        window5.append(tokens[i])
    return window5
    
def get_vector_bert(text, tokenizer_bert, model_bert):
    '''
    This function provides BERT embedding for the tweet.
    
    It uses "tensorflow" library as tf.
    
    Input: - text: a string 
           - tokenizer_bert: preloaded with transformers tokenizer
           - model_bert: preloaded with transformers model
    Output: 768-dimentional vector as list
    '''
    marked_text = "[CLS] " + text + " [SEP]"
    # Tokenize tweet with the BERT tokenizer
    tokenized_text = tokenizer_bert.tokenize(marked_text)
    indexed_tokens = tokenizer_bert.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Make vector 
    with torch.no_grad():
        outputs = model_bert(tokens_tensor, segments_tensors)
        #print(len(outputs))
        hidden_states = outputs[1]
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding.cpu().detach().numpy()
    
def get_vector_TFdistilbert(text, tokenizer_bert, model_bert):
    '''
    This function provides DistilBERT embedding for the text.
    
    It uses "torch" library to work with tensors.
    
    Input: - text: a string 
           - tokenizer_bert: preloaded with transformers tokenizer
           - model_bert: preloaded with transformers model
    Output: 768-dimentional vector as list
    '''
    marked_text = "[CLS] " + text + " [SEP]"
    # Tokenize tweet with the BERT tokenizer
    tokenized_text = tokenizer_bert.tokenize(marked_text)
    indexed_tokens = tokenizer_bert.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = tf.constant([indexed_tokens])
    segments_tensors = tf.constant([segments_ids])
    # Make vector 
    with torch.no_grad():
        outputs = model_bert(tokens_tensor)
        #print(outputs)
        hidden_states = list(outputs[0])
    token_vecs = hidden_states[0]
    #print(len(token_vecs))
    sentence_embedding = tf.math.reduce_mean(token_vecs, axis=0)
    return sentence_embedding
    
def get_vector_TFdistilbert_tokens(tokens, tokenizer_bert, model_bert):
    '''
    This function provides DistilBERT embedding for the list of tokens.

    Input: - tokens: a list of strings
           - tokenizer_bert: preloaded with transformers tokenizer
           - model_bert: preloaded with transformers model
    Output: 768-dimentional vector as list
    '''
    mean_vec = []
    for text in tokens:
        mean_vec.append(get_vector_TFdistilbert(text, tokenizer_bert, model_bert))
    return np.mean(mean_vec, axis = 0)