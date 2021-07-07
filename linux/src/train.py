import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import gc
import re
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
from split_data import hk_time
import datetime

#set seed
np.random.seed(4011)
#read data
json_path = os.path.abspath(os.path.join(__file__, '..', 'json'))
pic_path = os.path.abspath(os.path.join(__file__, '..', 'pic'))

train_df = pd.read_json(os.path.join(json_path, 'train.json'))
train_df = train_df.reset_index()

pred_df = pd.read_json(os.path.join(json_path, 'predict.json'))
pred_df = pred_df.reset_index()

test_df = pd.read_json(os.path.join(json_path, 'test.json'))
test_df = test_df.reset_index()
train_df = hk_time(train_df)
test_df = hk_time(test_df)
pred_df = hk_time(pred_df)


# split training data to validation
train_df, val_df = train_test_split(train_df, train_size=0.9, random_state=235)


print("train_df shape= ", train_df.shape)

#fill na
print('fill missing and get the values')
# fill missing and get the values

def data_col(col_name):
    X_train = train_df['Abstract'].fillna("na_").values
    X_val = val_df['Abstract'].fillna("na_").values
    X_test = test_df['Abstract'].fillna("na_").values
    y_train = train_df[col_name].values
    y_val = val_df[col_name].values
    y_test = test_df[col_name].fillna("na_").values
    X_pred = pred_df['Abstract'].fillna("na_").values
    y_pred = pred_df[col_name].values
    return X_train, X_val, X_test, y_train, y_val, y_test, X_pred, y_pred


#Change this parameter to get the result of different columns
X_train, X_val, X_test, y_train, y_val, y_test, X_pred, y_pred = data_col('neutral')

#Prepare Vectors For XGboost input

char_vector = TfidfVectorizer(
    ngram_range=(2,4),     
    max_features=20000,
    stop_words='english',   
    analyzer='char_wb',
    token_pattern=r'\w{1,}',
    strip_accents='unicode',
    sublinear_tf=True, 
    max_df=0.98,
    min_df=2  
)

print('fit char vector')
char_vector.fit(X_train[:8000])
print("fit success !")

print('transfer data based on char vector')
print('transfer train')
# return tfidf matrix
train_char_vector = char_vector.transform(X_train).tocsr() 
print('transfer validation')
valid_char_vector = char_vector.transform(X_val).tocsr()
print('transfer test')
test_char_vector = char_vector.transform(X_test).tocsr()
print('transfer predict')
pred_char_vector = char_vector.transform(X_pred).tocsr()
print("finished !")


all_text = list(X_train) + list(X_test) #without val and pred set


word_vector = TfidfVectorizer(
    ngram_range=(1,1),  # apply TFIDF to all words
    max_features=9000,
    sublinear_tf=True, 
    strip_accents='unicode', 
    analyzer='word', 
    token_pattern="\w{1,}", 
    stop_words="english",
    max_df=0.95,
    min_df=2
)


print('fit word vector')
word_vector.fit(all_text)
print("finished!")


print('transfer data based on word vector')
# transform to csr
train_word_vector = word_vector.transform(X_train).tocsr()
valid_word_vector = word_vector.transform(X_val).tocsr()
test_word_vector = word_vector.transform(X_test).tocsr()
pred_word_vector = word_vector.transform(X_pred).tocsr()
print("finished!")


#Features Engineering
data = [train_df, val_df, test_df, pred_df]
print("finished!")
mistake_list = ['colour', 'centre', 'favourite', 'travelling', 'counselling', 'theatre', 'cancelled', 'labour', 'organisation', 'wwii', 'citicise', 'youtu ', 'youtube ', 'Qoura', 'sallary', 'Whta', 'narcisist', 'howdo', 'whatare', 'howcan', 'howmuch', 'howmany', 'whydo', 'doI', 'theBest', 'howdoes', 'mastrubation', 'mastrubate', "mastrubating", 'pennis', 'Etherium', 'narcissit', 'bigdata', '2k17', '2k18', 'qouta', 'exboyfriend', 'airhostess', 'whst', 'watsapp', 'demonitisation', 'demonitization', 'demonetisation']

def get_features(data):
    # data = [train_df, val_df, test_df]  (3,)
    for dataframe in data:
        # dataFrame 
        dataframe["text_size"] = dataframe["Abstract"].apply(len).astype('uint16')  # text_length
        dataframe["capital_size"] = dataframe["Abstract"].apply(lambda x: sum(1 for c in x if c.isupper())).astype('uint16')  # number of Capital letter
        dataframe["capital_rate"] = dataframe.apply(lambda x: float(x["capital_size"]) / float(x["text_size"]), axis=1).astype('float16')  # rate of cap letter
        dataframe["exc_count"] = dataframe["Abstract"].apply(lambda x: x.count("!")).astype('uint16')  # # of !
        dataframe["quetion_count"] = dataframe["Abstract"].apply(lambda x: x.count("?")).astype('uint16')  # # of ?
        dataframe["unq_punctuation_count"] = dataframe["Abstract"].apply(lambda x: sum(x.count(p) for p in '∞θ÷α•à−β∅³π‘₹´°£€\×™√²')).astype('uint16') # # of $%^&
        dataframe["punctuation_count"] = dataframe["Abstract"].apply(lambda x: sum(x.count(p) for p in '.,;:^_`')).astype('uint16')  
        dataframe["symbol_count"] = dataframe["Abstract"].apply(lambda x: sum(x.count(p) for p in '*&$%')).astype('uint16')  
        dataframe["words_count"] = dataframe["Abstract"].apply(lambda x: len(x.split())).astype('uint16')  
        dataframe["unique_words"] = dataframe["Abstract"].apply(lambda x: (len(set(1 for w in x.split())))).astype('uint16')  
        dataframe["unique_rate"] = dataframe["unique_words"] / dataframe["words_count"]  
        dataframe["word_max_length"] = dataframe["Abstract"].apply(lambda x: max([len(word) for word in x.split()]) ).astype('uint16')
        dataframe["mistake_count"] = dataframe["Abstract"].apply(lambda x: sum(x.count(w) for w in mistake_list)).astype('uint16')  
    print("data shape = ", np.array(data).shape)
    return data

print('generate the features')
# data = [train_df, val_df, test_df]
data = get_features(data)
# print("data shape = ", np.array(data).shape)
# print(data)
print("finished!")

feature_cols = ["text_size", "capital_size", "capital_rate", "exc_count", "quetion_count", "unq_punctuation_count", "punctuation_count", "symbol_count", "words_count", "unique_words", "unique_rate", "word_max_length", "mistake_count"]


#Input Final Format
print('final preparation for input')

X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values
X_test = test_df[feature_cols].values
X_pred = pred_df[feature_cols].values


input_train = hstack([X_train, train_word_vector, train_char_vector])
input_valid = hstack([X_val, valid_word_vector, valid_char_vector])
input_test = hstack([X_test, test_word_vector, test_char_vector])
input_pred = hstack([X_pred, pred_word_vector, pred_char_vector])

#print('input_train: ', input_train)
train_word_vector = None
train_char_vector = None
valid_word_vector = None
valid_char_vector = None
test_word_vector = None
test_char_vector = None
pred_word_vector = None
pred_char_vector = None
#print('input_train: ', input_train)
print("finished!")

#Build The model
def build_xgb(train_X, train_y, valid_X, valid_y=None, subsample=0.75):

    xgtrain = xgb.DMatrix(train_X, label=train_y)
    if valid_y is not None:
        xgvalid = xgb.DMatrix(valid_X, label=valid_y)
    else:
        xgvalid = None
    
    model_params = {}
    # binary 0 or 1
    model_params['objective'] = 'reg:squarederror'
    # eta is the learning_rate, [default=0.3]
    model_params['eta'] = 0.3
    # depth of the tree, deeper more complex.
    model_params['max_depth'] = 6
    # 0 [default] print running messages, 1 means silent mode
    model_params['silent'] = 1
    model_params['eval_metric'] = 'rmse'
    # will give up further partitioning [default=1]
    model_params['min_child_weight'] = 1
    # subsample ratio for the training instance
    model_params['subsample'] = subsample
    # subsample ratio of columns when constructing each tree
    model_params['colsample_bytree'] = subsample
    # random seed
    model_params['seed'] = 4011
    # imbalance data ratio
    #model_params['scale_pos_weight'] = 
    
    # convert params to list
    model_params = list(model_params.items())
    
    return xgtrain, xgvalid, model_params

#Train The Model
def train_xgboost(xgtrain, xgvalid, model_params, num_rounds=500, patience=20):
    
    if xgvalid is not None:
        # watchlist what information should be printed. specify validation monitoring
        watchlist = [ (xgtrain, 'train'), (xgvalid, 'test') ]
        #early_stopping_rounds = stop if performance does not improve for k rounds
        model = xgb.train(model_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=patience)
    else:
        model = xgb.train(model_params, xgtrain, num_rounds)
    
    return model


input_train = csr_matrix(input_train)
input_valid = csr_matrix(input_valid)
input_test = csr_matrix(input_test)
input_pred = csr_matrix(input_pred)
print('train the model')
xgtrain, xgvalid, model_params = build_xgb(input_train, y_train ,input_valid, y_val)
model = train_xgboost(xgtrain, xgvalid, model_params)
print("finished!")

#Predict And Export Results
print('predict validation RMSE')
validate_hat = np.zeros(( X_val.shape[0], 1) )
validate_hat[:,0] = model.predict(xgb.DMatrix(input_valid), ntree_limit=model.best_ntree_limit)

validate_hat = list(validate_hat)
validate_hat = [float(x) for x in validate_hat]
y_val_list = [float(x) for x in y_val]
validate_hat = np.array(validate_hat)
y_val_list = np.array(y_val_list)
#RMSE/M
print(((sum((validate_hat- y_val_list)**2)//len(y_val_list))**0.5))

print('predict results RMSE')
predictions = np.zeros(( X_test.shape[0], 1) )
predictions[:,0] = model.predict(xgb.DMatrix(input_test), ntree_limit=model.best_ntree_limit)

predictions = list(predictions)
predictions = [float(x) for x in predictions]
y_test_list = [float(x) for x in y_test]
predictions = np.array(predictions)
y_test_list = np.array(y_test_list)
print(((sum((predictions- y_test_list)**2)//len(y_test_list))**0.5))
#sum The result into days

def sum_day(test_df):
    result_df = pd.concat([test_df['Releasing time'], pd.Series(predictions), pd.Series(y_test_list)], axis = 1)
    result_df.columns = ['date', 'predict', 'actual']
    result_df = result_df.sort_values(by=['date'])


    predicted_positive = result_df['predict'].groupby(result_df['date'].dt.to_period('d')).sum()
    predicted_positive = predicted_positive.reset_index()

    actual_positive = result_df['actual'].groupby(result_df['date'].dt.to_period('d')).sum()

    actual_positive = actual_positive.reset_index()

    result_df = pd.merge(predicted_positive, actual_positive, right_index = True, left_index = True) 
    result_df = result_df.drop(columns  = 'date_y')

    return result_df

#predict the actual result
print('predict 14 days result RMSE')
predictions_14 = np.zeros(( X_pred.shape[0], 1) )
predictions_14[:,0] = model.predict(xgb.DMatrix(input_pred), ntree_limit=model.best_ntree_limit)
predictions_14 = list(predictions_14)
predictions_14 = [float(x) for x in predictions_14]
y_pred_list = [float(x) for x in y_pred]
predictions_14 = np.array(predictions_14)
y_pred_list = np.array(y_pred_list)
print(((sum((predictions_14- y_pred_list)**2)//len(y_pred_list))**0.5))

result_df = sum_day(pred_df)


plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.style.use('ggplot')
plt.plot(result_df['date_x'].dt.to_timestamp(), result_df['predict'], linewidth=0.5)
plt.plot(result_df['date_x'].dt.to_timestamp(), result_df['actual'], linewidth=0.5)
plt.legend(['predicted result', 'actual result'], loc='upper left')
plt.savefig(os.path.join(pic_path, 'neutral.png'), dpi = 500)


pd.DataFrame(result_df['predict']).to_json(os.path.join(json_path, 'neutral.json'))
pd.DataFrame(result_df['date_x'].dt.to_timestamp()).to_json(os.path.join(json_path, 'date.json'))