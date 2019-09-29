import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import pairwise_distances
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# import dataset
df = pd.read_excel('FY19 Danceware Rating with New Codes.xlsx')

#Create user-item matrix
pivot = (df
          .groupby(['Customer', 'Product'])['Ratings']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Customer'))
matrix = np.array(pivot.values)

# Check sparsity
sparsity = float(len(matrix.nonzero()[0]))
sparsity /= (matrix.shape[0] * matrix.shape[1])
sparsity *= 100
print ('Sparsity: {:4.2f}%'.format(sparsity))

#Split dataset into training set and test set
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    #Remove users who bought 1 products from user-item matrix and paste to test set.
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=1, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test
train, test = train_test_split(matrix)

def sparsity(data):
    sparsity = float(len(data.nonzero()[0]))
    sparsity /= (data.shape[0] * data.shape[1])
    sparsity *= 100
    return sparsity
print ('training set sparsity: {:4.2f}%, test set sparsity: {:4.2f}%'.format(sparsity(train),sparsity(test)))

#Calculate the prediction by considering the top k users who are most similar to the input user (or, similarly, the top k items)
def predict_topK(ratings, similarity, kind='user', k=4):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    
    return pred
    
# Calculate user-user and item-item similarity matix using 'Pearson Correlation'
item_correlation = 1 - pairwise_distances(train.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0.
user_correlation = 1 - pairwise_distances(train, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0.

# Calculate predictions based on 'Pearson Correlation' and evaluate
pred_item2 = predict_topK(train, item_correlation, kind='item', k=10)
# pred_user2 = predict_topK(train, user_correlation, kind='user', k=10)
print ('Top-k Item-based CF MAE: ' + str(get_mae(test, pred_item2)))
# print ('Top-k User-based CF MAE: ' + str(get_mae(test, pred_user2)))

# Evaluations on predictions
def get_mae(true, pred):
    #Ignore zero terms.
    pred = pred[true.nonzero()].flatten()
    true = true[true.nonzero()].flatten()
    return mean_absolute_error(true, pred)
print ('Top-k Item-based CF MAE: ' + str(get_mae(test, pred_item)))
print ('Top-k User-based CF MAE: ' + str(get_mae(test, pred_user)))




# tune the value of K
k_array = [2, 4, 8, 16]
user_train_mae = []
user_test_mae = []
item_test_mae = []
item_train_mae = []

for k in k_array:
    user_pred_t = predict_topK(train, user_correlation, kind='user', k=k)
    item_pred_t = predict_topK(train, item_correlation, kind='item', k=k)
    
    user_train_mae += [get_mae(train, user_pred_t)]
    user_test_mae += [get_mae(test, user_pred_t)]
    item_train_mae += [get_mae(train, item_pred_t)]
    item_test_mae += [get_mae(test, item_pred_t)]

    
pal = sns.color_palette("Set2", 2)

plt.figure(figsize=(8, 8))
plt.plot(k_array, user_train_mae, c=pal[0], label='User-based train', alpha=0.5, linewidth=5)
plt.plot(k_array, user_test_mae, c=pal[0], label='User-based test', linewidth=5)
plt.plot(k_array, item_train_mae, c=pal[0], label='Item-based train', linewidth=5)
plt.plot(k_array, item_test_mae, c=pal[1], label='Item-based test', linewidth=5)
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=16);
plt.yticks(fontsize=16);
plt.xlabel('k', fontsize=30);
plt.ylabel('MAE', fontsize=30);
    
# Application on item-based recommendations
productID = pivot.T.index.values # Get the index of products
def top_k_products(similarity, mapper, product_idx, k=10):
    return [mapper[x] for x in np.argsort(similarity[product_idx,:])[:-k-1:-1]]
idx = 1 # Input a prodoct index
products = top_k_products(item_correlation, productID, idx, k=10)
print(productID[idx],"Similar products:",products[1:])
data_correlation = 1 - pairwise_distances(matrix.T, metric='correlation')
data_correlation[np.isnan(data_correlation)] = 0.
outputs=[]
for idx in range(len(productID)):
    products = top_k_products(data_correlation, productID, idx, k=10)
    outputs.append(products)
    
# Application on user-based recommendations
def user_based_rec(user,k):    #user index: user; amount of items we want to recommend
    # return user-based predictions
    pred_user_recommends=predict_topK(train, user_correlation, kind='user', k=k)
    # return products index that the user bought already
    productId_bought =[]
    for i in range(len(matrix[user])):
        if matrix[user][i]!=0:
            productId_bought.append(i)
    # return the products index the prediction predicts.
    productId_predict =[]
    for j in range(len(pred_user_recommends[user])):
        if pred_user_recommends[user][j]!=0:
            productId_predict.append(j)
    # return the index of the products with highest predictions        
    productId_bought_predict=np.argsort(pred_user_recommends[user])[::-1][:k*2]
    # return the index of the products that predict for the user and the user haven't bought it before
    final=[]
    for h in productId_bought_predict:
        if h not in productId_bought:
            final.append(h)
    # Return the products the user bought and recommend to buy.
    products_bought=[]
    for l in productId_bought:
        products_bought.append(productID[l])
      
    products_recommend=[]       
    for m in final:
        products_recommend.append(productID[m])
    return "You may like:",products[:k]
