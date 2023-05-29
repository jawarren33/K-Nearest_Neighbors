
# In the cell below, I develop my own code for performing 5-fold nested cross-validation along with the implemenation of k-NN above. 
# 
# The code for nested cross-validation should invoke the kNN function (see above). The cross validation function is  as:
# 
#     accuracies_fold = myNestedCrossVal(X,y,5,list(range(1,11)),['euclidean','manhattan'],mySeed)
#     
# where X is the data matrix (containing all samples and features for each sample), 5 is the number of folds, y are the known output labels, ``list(range(1,11)`` evaluates the neighbour parameter from 1 to 10, and ``['euclidean','manhattan',...]`` evaluates the distances on the validation sets.  mySeed is simply a random seed to enable us to replicate the results.
# 
# 
# myNestedCrossVal code

def myNestedCrossVal(X,Y,fold,Klist,distancelist,mySeed):
    #randomly split data into folds
    np.random.seed(mySeed) 
    shuffler = np.random.permutation(X.shape[0])
    X_shuffled = CleanX(X)[shuffler]
    Y_shuffled = Y[shuffler]
    X_folds = np.array_split(X_shuffled,fold)
    Y_folds = np.array_split(Y_shuffled,fold)
    
    #create lists to store the accuracies, k, and distance
    acc=[]
    K=[]
    D=[]
    
    #iterate through the folds
    
    for i in range(fold):
        X_test =  X_folds[i]
        Y_test =  Y_folds[i]
        x = X_folds[:i]+X_folds[i+1:]
        y = Y_folds[:i]+Y_folds[i+1:]
        accuraciesE=[]
        accuraciesM=[]
        
        #iterate through variations of k 
        
        for k in Klist:
            avg_euc = []
            avg_man = []
            for j in range(len(x)):
                X_val =  x[j]
                Y_val =  y[j]
                x_train = np.concatenate(np.array(x[:j]+ x[j+1:]), axis=0)
                y_train = np.concatenate(np.array(y[:j]+ y[j+1:]), axis=0)
                currY_E=knn_pred(x_train,y_train,X_val,k,0)
                Accuracy_E =  accuracy_score(Y_val,currY_E)
                currY_M= knn_pred(x_train,y_train,X_val,k,1)
                Accuracy_M =  accuracy_score(Y_val,currY_M)
                avg_euc.append(Accuracy_E)
                avg_man.append(Accuracy_M)
                
                
            accuraciesE.append(sum(avg_euc)/len(avg_euc))
            accuraciesM.append(sum(avg_man)/len(avg_man))
        d=1
        if max( accuraciesE)> max( accuraciesM):
            d=0
        if d==0:   
            bestK =  accuraciesE.index(max( accuraciesE))+1
        else:
            bestK =  accuraciesM.index(max( accuraciesM))+1
        K.append(bestK)
        if d==0:
            D.append(distancelist[0])
        else :
            D.append(distancelist[1])
        X_train = np.concatenate(np.array(x), axis=0)
        Y_train = np.concatenate(np.array(y), axis=0)       
        currY=knn_pred(X_train,Y_train,X_test,bestK,d)
        Accuracy =  accuracy_score(Y_test,currY)
        acc.append(Accuracy)
    return acc,K ,D 
                
# evaluate clean data code
myNestedCrossVal(X,y,5,list(range(1,11)),['euclidean','manhattan'],1)

# evaluate noisy  data code
myNestedCrossVal(XN,y,5,list(range(1,11)),['euclidean','manhattan'],1)

