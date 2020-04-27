import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import dataliner.preprocessing as dp


def test_drop_columns():
    df = pd.read_csv('titanic_train.csv')
    drop_columns_candidates = ['PassengerId',
                               ['PassengerId'],
                               ['PassengerId', 'Age']
                              ]

    for drop_columns in drop_columns_candidates:
        trans = dp.DropColumns(drop_columns=drop_columns)

        df_trans = trans.fit_transform(df)

        for col in drop_columns:
            assert col not in df_trans.columns
        assert df.shape[0] == df_trans.shape[0]


def test_drop_no_variance():
    df = pd.read_csv('titanic_train.csv')
    df['Test_0'] = 1
    df['Test_1'] = 'egg'

    trans = dp.DropNoVariance()

    df_trans = trans.fit_transform(df)

    assert 'Test_0' not in df_trans.columns
    assert 'Test_1' not in df_trans.columns
    assert df.shape[0] == df_trans.shape[0]


def test_drop_high_cardinality():
    df = pd.read_csv('titanic_train.csv')

    trans = dp.DropHighCardinality()

    df_trans = trans.fit_transform(df)
    assert df_trans.shape[1] == 6
    assert df.shape[0] == df_trans.shape[0]

    trans = dp.DropHighCardinality(max_categories=3)
    df_trans = trans.fit_transform(df)
    
    assert df_trans.shape[1] == 2
    assert df.shape[0] == df_trans.shape[0]


def test_drop_low_auc():
    df = pd.read_csv('titanic_train.csv')

    trans = dp.DropLowAUC(threshold=0.65)

    y = df['Survived']
    X = df.drop('Survived', axis=1)

    df_trans = trans.fit_transform(X, y)

    assert df_trans.shape[1] == 3
    assert df.shape[0] == df_trans.shape[0]

    trans = dp.DropLowAUC(threshold=0.7)

    df_trans = trans.fit_transform(X, y)

    assert df_trans.columns[0] == 'Sex'
    assert df.shape[0] == df_trans.shape[0]


def test_drop_high_correlation():
    df = pd.read_csv('titanic_train.csv')
    trans = dp.DropHighCorrelation(threshold=0.5)

    y = df['Survived']
    X = df.drop('Survived', axis=1)

    X['Age_copy'] = X['Age']
    X['Age_copy2'] = X['Age']
    X['Age_copy3'] = X['Age'] + 2
    X['Pclass_copy'] = X['Pclass']

    X_trans = trans.fit_transform(X, y)

    assert X_trans.shape[1] == 10
    assert X.shape[0] == X_trans.shape[0]


def test_impute_nan():
    df = pd.read_csv('titanic_train.csv')
    trans = dp.ImputeNaN()

    df_trans = trans.fit_transform(df)

    assert df_trans.isnull().sum().sum() == 0
    assert df.shape[0] == df_trans.shape[0]


def test_one_hot_encoding():
    df = pd.read_csv('titanic_train.csv')
    trans = dp.OneHotEncoding()

    y = df['Survived']
    X = df.drop('Survived', axis=1)

    df_trans = trans.fit_transform(df)

    assert df_trans.shape[1] == 1726
    assert df.shape[0] == df_trans.shape[0]


def test_binalize_nan():
    df = pd.read_csv('titanic_train.csv')
    trans = dp.BinarizeNaN()

    y = df['Survived']
    X = df.drop('Survived', axis=1)

    df_trans = trans.fit_transform(df)

    assert df['Age'].isna().sum() == df_trans['Age_NaNFlag'].sum()
    assert df['Cabin'].isna().sum() == df_trans['Cabin_NaNFlag'].sum()
    assert df['Embarked'].isna().sum() == df_trans['Embarked_NaNFlag'].sum()
    assert df.shape[0] == df_trans.shape[0]


def test_count_row_nan():
    df = pd.read_csv('titanic_train.csv')
    trans = dp.CountRowNaN()

    y = df['Survived']
    X = df.drop('Survived', axis=1)

    df_trans = trans.fit_transform(df)

    assert df.isna().sum().sum() == df_trans['NaN_Totals'].sum()
    assert df.shape[0] == df_trans.shape[0]


def test_standardize_data():
    df = pd.read_csv('titanic_train.csv')
    trans = dp.StandardizeData()

    y = df['Survived']
    X = df.drop('Survived', axis=1)

    df_trans = trans.fit_transform(df)

    assert df.shape[0] == df_trans.shape[0]
    assert df.shape[1] == df_trans.shape[1]
    assert df_trans['PassengerId'][0] == (df['PassengerId'][0]
            - df['PassengerId'].mean())/df['PassengerId'].std()

    X['Test'] = 0.0
    Xt = trans.fit_transform(X)

    assert Xt['Test'].unique()[0] == 0


def test_clip_data():
    df = pd.read_csv('titanic_train.csv')
    trans = dp.ClipData()

    y = df['Survived']
    X = df.drop('Survived', axis=1)
    X['Test'] = 0

    df_trans = trans.fit_transform(df)

    for feature in ['Age', 'Fare']:
        lowerbound = min(np.percentile(df[feature].dropna(), [1, 99]))
        upperbound = max(np.percentile(df[feature].dropna(), [1, 99]))
        assert df_trans[feature].dropna().min() == lowerbound
        assert df_trans[feature].dropna().max() == upperbound
    assert df.shape[0] == df_trans.shape[0]
    assert df.shape[1] == df_trans.shape[1]


def test_group_rare_category():
    df = pd.read_csv('titanic_train.csv')
    trans = dp.GroupRareCategory(threshold=0.01)

    y = df['Survived']
    X = df.drop('Survived', axis=1)
    X['Test'] = 0

    df_trans = trans.fit_transform(df)
    sample = df['Cabin'].value_counts(ascending=False) 

    cats_df = sample[sample > sample.sum()*0.01].index
    cats_df_trans = df_trans['Cabin'].value_counts(ascending=False).index
    dummy_string = np.setdiff1d(cats_df_trans, cats_df)

    assert (sample <= sample.sum()*0.01).count() == 147
    assert df.shape[0] == df_trans.shape[0]
    assert df.shape[1] == df_trans.shape[1]
    assert dummy_string == 'RareCategory'


def test_target_mean_encoding():
    df = pd.read_csv('titanic_train.csv')
    trans = dp.TargetMeanEncoding()

    y = df['Survived']
    X = df.drop('Survived', axis=1)

    df_trans = trans.fit_transform(X, y)

    assert df_trans['Name'].mean() == 0.38383838383838975
    assert df_trans['Sex'].mean() == 0.38383838383838054
    assert df_trans['Cabin'].mean() == 0.35791513764516214
    assert df_trans['Ticket'].mean() == 0.4306411823436723
    assert df_trans['Embarked'].mean() == 0.38367351680115463
    assert df.shape[0] == df_trans.shape[0]
    assert df.shape[1] == df_trans.shape[1] + 1