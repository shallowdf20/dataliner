import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

PACKAGE_TEST = False
if PACKAGE_TEST:
    import dataliner as dp
else:
    import sys
    sys.path.append('../dataliner')
    import preprocessing as dp


TRAIN_DATA = 'titanic_train.csv'
TEST_DATA = 'titanic_test.csv'
TARGET_NAME = 'Survived'


def _setup():
    df = pd.read_csv(TRAIN_DATA)
    X_test = pd.read_csv(TEST_DATA)
    X = df.drop(TARGET_NAME, axis=1)
    y = df[TARGET_NAME]
    return X, X_test, y


def _check_equal_rows(df1, df2):
    assert df1.shape[0] == df2.shape[0]


def _check_equal_cols(df1, df2):
    assert df1.shape[1] == df2.shape[1]


def _check_col_does_not_exist_in_df(df, col):
    assert col not in df.columns


def _check_col_exist_in_df(df, col):
    assert col in df.columns


def _check_number_of_cols_equal(df, num):
    assert df.shape[1] == num


def _check_same_cols_and_order(df1, df2):
    assert np.array_equal(df1.columns, df2.columns)


def test_drop_columns():
    X, X_test, _ = _setup()

    drop_columns_candidates = [
        'PassengerId', ['PassengerId'], ['PassengerId', 'Age']
    ]

    for drop_columns in drop_columns_candidates:
        trans = dp.DropColumns(drop_columns=drop_columns)
        Xt = trans.fit_transform(X)
        Xt_test = trans.transform(X_test)

        _check_equal_rows(X, Xt)
        _check_equal_rows(X_test, Xt_test)
        
        for col in drop_columns:
            _check_col_does_not_exist_in_df(Xt, col)
            _check_col_does_not_exist_in_df(Xt_test, col)
        
        _check_same_cols_and_order(Xt, Xt_test)


def test_drop_no_variance():
    X, X_test, _ = _setup()

    X['Test_0'] = 1
    X['Test_1'] = 'egg'
    trans = dp.DropNoVariance()
    Xt = trans.fit_transform(X)
    _check_col_does_not_exist_in_df(Xt, 'Test_0')
    _check_col_does_not_exist_in_df(Xt, 'Test_1')
    _check_equal_rows(X, Xt)
    
    X_test['Test_0'] = 1
    X_test['Test_1'] = 'egg'
    Xt_test = trans.transform(X_test)
    _check_col_does_not_exist_in_df(Xt_test, 'Test_0')
    _check_col_does_not_exist_in_df(Xt_test, 'Test_1')
    _check_equal_rows(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_drop_high_cardinality():
    X, X_test, _ = _setup()

    trans = dp.DropHighCardinality()
    Xt = trans.fit_transform(X)
    _check_number_of_cols_equal(Xt, 8)
    _check_equal_rows(X, Xt)

    trans = dp.DropHighCardinality(max_categories=3)
    Xt = trans.fit_transform(X)
    _check_number_of_cols_equal(Xt, 7)
    _check_equal_rows(X, Xt)
    
    Xt_test = trans.transform(X_test)
    _check_number_of_cols_equal(Xt_test, 7)
    _check_equal_rows(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_drop_low_auc():
    X, X_test, y = _setup()

    trans = dp.DropLowAUC(threshold=0.65)
    Xt = trans.fit_transform(X, y)
    _check_number_of_cols_equal(Xt, 3)
    _check_equal_rows(X, Xt)

    trans = dp.DropLowAUC(threshold=0.7)
    Xt = trans.fit_transform(X, y)
    assert Xt.columns[0] == 'Sex'
    _check_number_of_cols_equal(Xt, 1)
    _check_equal_rows(X, Xt)
    
    Xt_test = trans.transform(X_test)
    assert Xt_test.columns[0] == 'Sex'
    _check_number_of_cols_equal(Xt, 1)
    _check_equal_rows(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_drop_high_correlation():
    X, X_test, y = _setup()
    trans = dp.DropHighCorrelation(threshold=0.5)

    X['Age_copy'] = X['Age']
    X['Age_copy2'] = X['Age']
    X['Age_copy3'] = X['Age'] + 2
    X['Pclass_copy'] = X['Pclass']

    Xt = trans.fit_transform(X, y)
    _check_number_of_cols_equal(Xt, 10)
    _check_equal_rows(X, Xt)
    
    Xt_test = trans.transform(X_test)
    _check_number_of_cols_equal(Xt_test, 10)
    _check_equal_rows(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_impute_nan():
    X, X_test, _ = _setup()
    trans = dp.ImputeNaN()

    Xt = trans.fit_transform(X)

    assert Xt.isnull().sum().sum() == 0
    _check_equal_rows(X, Xt)
    
    Xt_test = trans.transform(X_test)
    assert Xt_test.isnull().sum().sum() == 0
    _check_equal_rows(X_test, Xt_test)
    
    _check_same_cols_and_order(Xt, Xt_test)


def test_one_hot_encoding():
    X, X_test, _ = _setup()
    trans = dp.OneHotEncoding()

    Xt = trans.fit_transform(X)

    _check_number_of_cols_equal(Xt, 1725)
    _check_equal_rows(X, Xt)
    
    Xt_test = trans.transform(X_test)
    _check_number_of_cols_equal(Xt_test, 1725)
    _check_equal_rows(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_binalize_nan():
    X, X_test, _ = _setup()
    trans = dp.BinarizeNaN()
    Xt = trans.fit_transform(X)

    assert X['Age'].isna().sum() == Xt['Age_NaNFlag'].sum()
    assert X['Cabin'].isna().sum() == Xt['Cabin_NaNFlag'].sum()
    assert X['Embarked'].isna().sum() == Xt['Embarked_NaNFlag'].sum()
    _check_equal_rows(X, Xt)
    
    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_count_row_nan():
    X, X_test, _ = _setup()
    trans = dp.CountRowNaN()
    Xt = trans.fit_transform(X)

    assert X.isna().sum().sum() == Xt['NaN_Totals'].sum()
    _check_equal_rows(X, Xt)
    
    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_clip_data():
    X, X_test, _ = _setup()
    trans = dp.ClipData()

    Xt = trans.fit_transform(X)

    for feature in ['Age', 'Fare']:
        lowerbound = min(np.percentile(X[feature].dropna(), [1, 99]))
        upperbound = max(np.percentile(X[feature].dropna(), [1, 99]))
        assert Xt[feature].dropna().min() == lowerbound
        assert Xt[feature].dropna().max() == upperbound
    _check_equal_rows(X, Xt)
    _check_equal_cols(X, Xt)
    
    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)
    _check_equal_cols(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_group_rare_category():
    X, X_test, _ = _setup()
    threshold = 0.01
    trans = dp.GroupRareCategory(threshold=threshold)

    Xt = trans.fit_transform(X)
    sample = X['Cabin'].value_counts(ascending=False) 

    cats_df = sample[sample > sample.sum()*threshold].index
    cats_df_trans = Xt['Cabin'].value_counts(ascending=False).index
    dummy_string = np.setdiff1d(cats_df_trans, cats_df)

    assert (sample <= sample.sum()*0.01).count() == 147
    _check_equal_rows(X, Xt)
    _check_equal_cols(X, Xt)
    assert dummy_string == 'RareCategory'
    
    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)
    _check_equal_cols(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_target_mean_encoding():
    X, X_test, y = _setup()
    trans = dp.TargetMeanEncoding()

    Xt = trans.fit_transform(X, y)

    assert Xt['Name'].mean() == 0.38383838383838975
    assert Xt['Sex'].mean() == 0.38383838383838054
    assert Xt['Cabin'].mean() == 0.35791513764516214
    assert Xt['Ticket'].mean() == 0.4306411823436723
    assert Xt['Embarked'].mean() == 0.38367351680115463
    _check_equal_rows(X, Xt)
    _check_equal_cols(X, Xt)
    
    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)
    _check_equal_cols(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_standard_scaling():
    X, X_test, _ = _setup()
    trans = dp.StandardScaling()

    Xt = trans.fit_transform(X)

    _check_equal_rows(X, Xt)
    _check_equal_cols(X, Xt)
    assert Xt['PassengerId'][0] == (X['PassengerId'][0]
            - X['PassengerId'].mean())/X['PassengerId'].std()

    X['Test'] = 0.0
    Xt = trans.fit_transform(X)

    assert Xt['Test'].unique()[0] == 0
    assert Xt.isnull().sum().sum() == 866
    
    X_test['Test'] = 0.0
    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)
    _check_equal_cols(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_min_max_scaling():
    X, X_test, _ = _setup()
    trans = dp.MinMaxScaling()

    Xt = trans.fit_transform(X)

    _check_equal_rows(X, Xt)
    _check_equal_cols(X, Xt)
    assert Xt['PassengerId'][0] == (
        (X['PassengerId'][0] - X['PassengerId'].min()) /
        (X['PassengerId'].max() - X['PassengerId'].min())
        )
    assert Xt['PassengerId'].mean() == 0.499999999999997
    assert Xt['Age'].mean() == 0.36792055349407926
    assert Xt['Fare'].mean() == 0.06285842768394748
    assert Xt['Pclass'].mean() == 0.654320987654321


    X['Test'] = 0.0
    Xt = trans.fit_transform(X)

    assert Xt['Test'].unique()[0] == 0

    X_test['Test'] = 0.0
    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)
    _check_equal_cols(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_count_encoding():
    X, X_test, _ = _setup()
    trans = dp.CountEncoding()

    Xt = trans.fit_transform(X)

    _check_equal_rows(X, Xt)
    _check_equal_cols(X, Xt)

    assert Xt['Sex'][0] == 577
    assert Xt['Embarked'][0] == 644
    assert Xt['Cabin'][0] == 687

    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)
    _check_equal_cols(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_ranked_count_encoding():
    X, X_test, _ = _setup()
    trans = dp.RankedCountEncoding()

    Xt = trans.fit_transform(X)

    _check_equal_rows(X, Xt)
    _check_equal_cols(X, Xt)

    assert (Xt.groupby('Embarked').count()['PassengerId'].tolist()
            == [644, 168, 77, 2])
    assert (Xt.groupby('Sex').count()['PassengerId'].tolist()
            == [577, 314])
    
    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)
    _check_equal_cols(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_frequency_encoding():
    X, X_test, _ = _setup()
    trans = dp.FrequencyEncoding()

    Xt = trans.fit_transform(X)

    _check_equal_rows(X, Xt)
    _check_equal_cols(X, Xt)

    assert Xt['Sex'].unique().tolist() == [
            0.6475869809203143,
            0.35241301907968575]
    assert Xt['Embarked'].unique().tolist() == [
            0.7227833894500562,
            0.18855218855218855,
            0.08641975308641975,
            0.002244668911335578]
    assert Xt['Cabin'].unique().tolist() == [
            0.7710437710437711,
            0.001122334455667789,
            0.002244668911335578,
            0.004489337822671156,
            0.003367003367003367]

    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)
    _check_equal_cols(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_ranked_target_mean_encoding():
    X, X_test, y = _setup()
    trans = dp.RankedTargetMeanEncoding()
    Xt = trans.fit_transform(X, y)

    _check_equal_rows(X, Xt)
    _check_equal_cols(X, Xt)

    assert Xt['Embarked'].mean() == 3.529741863075196
    assert Xt['Sex'].mean() == 1.6475869809203143
    assert Xt['Cabin'].mean() == 126.05387205387206

    Xt_test = trans.transform(X_test)
    _check_equal_rows(X_test, Xt_test)
    _check_equal_cols(X_test, Xt_test)

    _check_same_cols_and_order(Xt, Xt_test)


def test_append_anomaly_score():
    X, X_test, y = _setup()
    col_y = 'Survived'

    trans2 = dp.RankedTargetMeanEncoding()
    trans = dp.AppendAnomalyScore()

    Xt = trans2.fit_transform(X, y)
    Xt = trans.fit_transform(Xt.fillna(0))

    assert Xt['Anomaly_Score'].mean() == 0.024687907895391693
    _check_equal_rows(X, Xt)
    _check_col_exist_in_df(Xt, 'Anomaly_Score')
    assert X.shape[1] == Xt.shape[1] - 1

    Xt_test = trans.transform((trans2.transform(X_test.fillna(0))))
    _check_equal_rows(X_test, Xt_test)
    _check_col_exist_in_df(Xt_test, 'Anomaly_Score')

    _check_same_cols_and_order(Xt, Xt_test)


def test_append_cluster():
    X, X_test, y = _setup()

    trans2 = dp.RankedTargetMeanEncoding()
    trans = dp.AppendCluster()

    Xt = trans2.fit_transform(X, y)
    Xt = trans.fit_transform(Xt.fillna(0))

    assert Xt['Cluster_Number'].mean() == 3.2435465768799103
    _check_equal_rows(X, Xt)
    assert X.shape[1] == Xt.shape[1] - 1

    Xt_test = trans.transform((trans2.transform(X_test.fillna(0))))
    _check_equal_rows(X_test, Xt_test)
    _check_col_exist_in_df(Xt_test, 'Cluster_Number')

    _check_same_cols_and_order(Xt, Xt_test)


def test_append_cluster_distance():
    X, X_test, y = _setup()

    trans2 = dp.RankedTargetMeanEncoding()
    trans = dp.AppendClusterDistance()

    Xt = trans2.fit_transform(X, y)
    Xt = trans.fit_transform(Xt.fillna(0))

    assert Xt['Cluster_Distance_0'].mean() == 530.2754787693156
    _check_equal_rows(X, Xt)
    assert X.shape[1] == Xt.shape[1] - 8
    
    Xt_test = trans.transform((trans2.transform(X_test.fillna(0))))
    _check_equal_rows(X_test, Xt_test)
    _check_col_exist_in_df(Xt_test, 'Cluster_Distance_0')

    _check_same_cols_and_order(Xt, Xt_test)


def test_append_principal_component():
    X, X_test, y = _setup()

    trans2 = dp.RankedTargetMeanEncoding()
    trans3 = dp.StandardScaling()
    trans = dp.AppendPrincipalComponent()

    Xt = trans2.fit_transform(X, y)
    Xt = trans3.fit_transform(Xt)
    Xt = trans.fit_transform(Xt.fillna(0))

    assert Xt['Principal_Component_0'].max() == 7.497300874940136
    _check_equal_rows(X, Xt)
    assert X.shape[1] == Xt.shape[1] - 5
    
    Xt_test = trans.transform((trans2.transform(X_test.fillna(0))))
    _check_equal_rows(X_test, Xt_test)
    _check_col_exist_in_df(Xt_test, 'Principal_Component_0')

    _check_same_cols_and_order(Xt, Xt_test)


def test_pipelines():
    X, X_test, y = _setup()

    ctrans_candidates = [
        dp.OneHotEncoding(),
        dp.TargetMeanEncoding(),
        dp.CountEncoding(),
        dp.RankedCountEncoding(),
        dp.FrequencyEncoding(),
        dp.RankedTargetMeanEncoding(),
    ]

    scaler_candidates = [
        dp.StandardScaling(),
        dp.MinMaxScaling()
    ]

    for scaler in scaler_candidates:
        for ctrans in ctrans_candidates:
            process = make_pipeline(
                dp.DropColumns(drop_columns="PassengerId"),
                dp.DropNoVariance(),
                dp.GroupRareCategory(),
                dp.ClipData(),
                dp.DropHighCardinality(),
                dp.BinarizeNaN(),
                dp.CountRowNaN(),
                dp.ImputeNaN(),
                ctrans,
                dp.DropNoVariance(),
                dp.DropHighCorrelation(),
                scaler,
                dp.AppendAnomalyScore(),
                dp.AppendCluster(),
                dp.AppendClusterDistance(),
                dp.AppendPrincipalComponent(),
                dp.DropHighCorrelation(),
                dp.DropLowAUC(),
            )

            Xt = process.fit_transform(X, y)
            Xt_test = process.transform(X_test)
            
            _check_equal_rows(X, Xt)
            _check_equal_rows(X_test, Xt_test)
            
            _check_same_cols_and_order(Xt, Xt_test)


def test_cascaded_encoders():
    X, X_test, y = _setup()

    process = make_pipeline(
        dp.ImputeNaN(),
        dp.OneHotEncoding(),
        dp.TargetMeanEncoding(),
        dp.CountEncoding(),
        dp.RankedCountEncoding(),
        dp.FrequencyEncoding(),
        dp.RankedTargetMeanEncoding(),
    )
    
    Xt = process.fit_transform(X, y)
    Xt_test = process.transform(X_test)
    
    _check_equal_rows(X, Xt)
    _check_equal_rows(X_test, Xt_test)
    
    _check_same_cols_and_order(Xt, Xt_test)

def test_arithmetic_feature_generator():
    X, X_test, y = _setup()

    operation_candidates = [
        'add',
        'subtract',
        'multiply',
        'divide'
    ]
    metric_candidates = ['roc_auc', 'accuracy']

    for metric in metric_candidates:
        for operation in operation_candidates:
            process = make_pipeline(
                dp.ImputeNaN(),
                dp.ArithmeticFeatureGenerator(metric=metric,
                                              operation=operation)
            )

            Xt = process.fit_transform(X, y)
            Xt_test = process.transform(X_test)
            
            _check_equal_rows(X, Xt)
            _check_equal_rows(X_test, Xt_test)
            
            _check_same_cols_and_order(Xt, Xt_test)
