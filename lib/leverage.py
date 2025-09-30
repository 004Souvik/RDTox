import pandas as pd
import numpy as np




def leverage_calculator(data1: pd.DataFrame, data2: pd.DataFrame):
    '''
    This function calculates the leverage of training and test sets. \n
    data1: training set descriptor matrix (n_samples x n_features) \n
    data2: test set descriptor matrix (m_samples x n_features)\n
    Returns:
        leverage_tr_df: Leverage values for training data \n
        leverage_te_df: Leverage values for test data 
    '''

    if data1.shape[1] != data2.shape[1]:
        raise ValueError("Input files must have the same number of columns/features.")
    
    p_val = len(data1.columns)
    n_val = len(data1)
    h_star = 3*(p_val+1)/(n_val)
    print(h_star)
    
    data1.insert(0, "dummey", 1)
    data2.insert(0, "dummey", 1)
    X = data1.values  # shape (n, p)
    X_test = data2.values  # shape (m, p)

    # (X^T X)^-1
    XtX_inv = np.linalg.inv(X.T @ X)

    # Leverage for training data: diag(X @ (X^T X)^-1 @ X^T)
    H_train = X @ XtX_inv @ X.T
    leverage_tr = np.diag(H_train)

    # Leverage for test data: row-wise x_i @ (X^T X)^-1 @ x_i^T
    leverage_te = np.array([x @ XtX_inv @ x.T for x in X_test])

    leverage_tr_df = pd.DataFrame(leverage_tr, columns=["Leverage Value"], index=data1.index)
    leverage_te_df = pd.DataFrame(leverage_te, columns=["Leverage Value"], index=data2.index)

    leverage_tr_df["AD Status"] = np.where(leverage_tr_df["Leverage Value"]>=h_star, "Outside AD", "Inside AD")
    leverage_te_df["AD Status"] = np.where(leverage_te_df["Leverage Value"]>=h_star, "Outside AD", "Inside AD")

    return leverage_tr_df, leverage_te_df, h_star

