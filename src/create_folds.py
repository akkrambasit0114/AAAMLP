# import pandas and model_selection module of scikit-learn

import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    # create relative folder based on project==>
    #f = open("C:\\Users\\Basit Akram\\Documents\\new-project\\input\\cat_train.csv") ##--VSCODE
    # you don't need to open the file explicitly just pass the file location to read_csv
    #df = pd.read_csv(f) ##-- VSCODE    
    # Training data is in a cvs file called train.csv
    df = pd.read_csv("C:\\Users\\Basit Akram\\Documents\\new-project\\input\\adult.csv")  
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    target="income"

    # fetch targets
    y = df[target].values
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    #print(kf)
    # fill the new kfold column
    #does this code works?? x should be df[except y]
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df,y=y)):
        #print(len(train_idx), len(val_idx))
        #print(fold, train_idx, val_idx)
        df.loc[val_idx, 'kfold'] = fold
        
    df.to_csv("C:\\Users\\Basit Akram\\Documents\\new-project\\input\\adult_folds.csv",index=False)
    