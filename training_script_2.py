import pickle
import pandas as pd
import numpy as np
import xgboost as xgb



if __name__ == '__main__':
    dataset = pickle.load(open('orientation_df_train', 'rb'))

    eval_metric = ["auc", "error"]
    classifier = xgb.XGBClassifier(silent=False,
                                   scale_pos_weight=1,
                                   learning_rate=0.1,
                                   colsample_bytree=0.8,
                                   subsample=0.8,
                                   objective='binary:logistic',
                                   n_estimators=1000,
                                   reg_alpha=0.3,
                                   max_depth=5,
                                   gamma=0.1)

    metrics = ['braycurtis']

    for metric in metrics:
        print('Creating dataset for {} metric'.format(metric))
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        
        print('Now training...')
        classifier.fit(X, y, eval_metric=eval_metric, verbose=True)
        print('Training finished.')

        pickle.dump(classifier, open('weights_xgb.data', 'wb'))
