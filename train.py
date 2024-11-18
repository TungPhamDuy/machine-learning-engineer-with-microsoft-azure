from sklearn.linear_model import LogisticRegression
import argparse
import os
import joblib

from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.run import Run
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np 


def clean_data(data):
    x_df = data.to_pandas_dataframe()
    # add defaulted column to the y_df
    y_df = x_df["defaulted"]
    # drop that column from x_df
    x_df.drop('defaulted', axis=1, inplace=True)
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse regularization. Smaller values -> stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number converge iterations")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization:", np.float(args.C))
    run.log("Max iters:", np.int(args.max_iter))

    # create TabularDataset using TabularDatasetFactory
    url = "https://raw.githubusercontent.com/TungPhamDuy/azmlproject3/refs/heads/main/credit-card-default.csv"
    ds = TabularDatasetFactory.from_delimited_files(path=url)
    
    x, y = clean_data(ds)

    # split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))
    run.log("Accuracy", np.float(accuracy))

    # save the model to the outputs folder for deployment
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model_'+'accuracy_'+str(accuracy)+"_"+'C_'+str(args.C)+"_"+'maxIter_'+str(args.max_iter)+'.joblib')

if __name__ == '__main__':
    main()
