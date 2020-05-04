# -*- coding: utf-8 -*-
"""
A dataprocessing package for data preprocess and feature engineering.

This library contains preprocessing methods for data processing
and feature engineering used during data analysis and machine learning 
process.
"""

import itertools
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted


__all__ = [
    'DropColumns',
    'DropNoVariance',
    'DropHighCardinality',
    'DropLowAUC',
    'DropHighCorrelation',
    'ImputeNaN',
    'OneHotEncoding',
    'BinarizeNaN',
    'CountRowNaN',
    'StandardizeData',
    'ClipData',
    'GroupRareCategory',
    'TargetMeanEncoding',
    'StandardScaling',
    'MinMaxScaling',
    'CountEncoding',
    'RankedCountEncoding',
    'FrequencyEncoding',
    'RankedTargetMeanEncoding',
    'AppendAnomalyScore',
    'AppendCluster',
    'AppendClusterDistance',
    'AppendPrincipalComponent',
    'ArithmeticFeatureGenerator', # from 1.2
    'RankedEvaluationMetricEncoding', # from 1.2
    'AppendClassificationModel', # from 1.2
    'AppendEncoder', # from 1.2
    'AppendClusterTargetMean', # from 1.2
    'PermutationImportanceTest' # from 1.2
]


def _check_X(X):
    if isinstance(X, pd.DataFrame):
        pass
    else:
        raise TypeError("Input X is not a pandas DataFrame.")


def _check_y(y):
    if isinstance(y, pd.Series):
        pass
    else:
        raise TypeError("Input y is not a pandas Series.")


def _check_X_y(X, y):
    _check_X(X)
    _check_y(y)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of rows are different between X and y.")


def _check_duplicate(list_):
    return len(list_) != len(set(list_))


def _check_fit(list1, list2):
    if set(list1) == set(list2):
        pass
    elif _check_duplicate(list1) or _check_duplicate(list2):
        raise ValueError("There are features with duplicate name.")
    else:
        raise ValueError("Columns are different from when fitted. For\
                preprocess with model transforming such as Isolation\
                Forest or KMeans, it require the columns to be same.")


def _check_binary(y):
    if len(y.unique()) == 2:
        pass
    else:
        raise Exception("This class can only be used for Binary Classification")


def _check_method_implemented(model, method_str):
    if method_str in dir(model):
        pass
    else:
        raise Exception(method_str + ' is not implemented in the specified model')


def load_titanic():
    """
    Load train and test data for titanic datasets.

    :return: train_features, train_target, test_features
    :rtype: pandas.DataFrame, pandas.Series, pandas.DataFrame
    """
    path = os.path.dirname(__file__)
    df = pd.read_csv(path + '/datasets/titanic_train.csv')
    X_test = pd.read_csv(path + '/datasets/titanic_test.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, X_test, y


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Simply delete columns specified from input dataframe.

    :param list drop_columns: List of feature names which will be droped \
    from input dataframe. For single columns, string can also be used.\
    (default=None)
    """

    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        """
        Fit transformer by checking X is a pandas DataFrame.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        if isinstance(self.drop_columns, list):
            for col in self.drop_columns:
                if col in X.columns:
                    pass
                else:
                    raise Exception("Specified columns are not in the input.")
        else:
            if self.drop_columns in X.columns:
                pass
            else:
                raise Exception("Specified column is not in the input.")

        self.is_fitted_ = X.columns.shape

        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        if self.drop_columns is None:
            return X
        else:
            Xt = X.drop(self.drop_columns, axis=1)
            return Xt


class DropNoVariance(BaseEstimator, TransformerMixin):
    """
    Delete columns which only have single unique value.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit transformer by deleting column with single unique value.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.drop_columns_ = None

        for feature in X.columns:
            if X[feature].unique().shape[0] == 1:
                if self.drop_columns_ is None:
                    self.drop_columns_ = [feature]
                else:
                    self.drop_columns_ = np.append(self.drop_columns_, feature)

        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        if self.drop_columns_ is not None:
            drop_columns = np.intersect1d(self.drop_columns_, X.columns)
        else:
            drop_columns = self.drop_columns_

        if drop_columns is None:
            return X
        else:
            Xt = X.drop(drop_columns, axis=1)
            return Xt


class DropHighCardinality(BaseEstimator, TransformerMixin):
    """
    Delete columns with high cardinality.
    Basically means dropping column with too many categories.

    :param int max_categories: Maximum number of categories to be permitted\
    in a column. If number of categories in a certain column exceeds this\
    value, that column will be deleted. (default=50)
    """

    def __init__(self, max_categories=50):
        self.max_categories = max_categories

    def fit(self, X, y=None):
        """
        Fit transformer by deleting column with high cardinality.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.drop_columns_ = None
        cat_columns = X.select_dtypes(exclude='number').columns

        for feature in cat_columns:
            if X[feature].unique().shape[0] >= self.max_categories:
                if self.drop_columns_ is None:
                    self.drop_columns_ = [feature]
                else:
                    self.drop_columns_ = np.append(self.drop_columns_, feature)

        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        if self.drop_columns_ is not None:
            drop_columns = np.intersect1d(self.drop_columns_, X.columns)
        else:
            drop_columns = self.drop_columns_

        if drop_columns is None:
            return X
        else:
            Xt = X.drop(drop_columns, axis=1)
            return Xt


class DropLowAUC(BaseEstimator, TransformerMixin):
    """
    Delete columns that have low information to predict target variable.\
    This class calculate roc_auc by fitting all features in the input\
    array one by one against target feature using Logistic Regression\
    and drop features with roc_auc below threshold specified.\
    Missing values will be replaced by mean and categorical feature \
    will be converted to dummy variables by one hot encoding with missing\
    values filled with mode.

    :param float threshold: Threshold value for roc_auc. Feature with roc_auc \
    below this value will be deleted. (default=0.51)
    """

    def __init__(self, threshold=0.51):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fit transformer by fitting each feature with Logistic \
        Regression and storing features with roc_auc less than threshold 

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Input Series for target variable
        :return: fitted object (self)
        :rtype: object
        """
        _check_X_y(X, y)
        _check_binary(y)

        self.drop_columns_ = None

        cv = StratifiedKFold(n_splits=5)
        lr = LogisticRegression(penalty='l2', solver='lbfgs')

        for feature in X.columns:
            X_lr = X[[feature]]
            if X_lr.dtypes[0] == 'object':
                mode = X_lr.mode()
                X_lr = X_lr.fillna(mode)
                X_lr = pd.get_dummies(X_lr)
            else:
                mean = X_lr.mean()
                X_lr = X_lr.fillna(mean)

            roc_auc = cross_val_score(lr, X_lr, y, cv=cv).mean()

            if roc_auc < self.threshold:
                if self.drop_columns_ is None:
                    self.drop_columns_ = [feature]
                else:
                    self.drop_columns_ = np.append(self.drop_columns_, feature)

        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        if self.drop_columns_ is not None:
            drop_columns = np.intersect1d(self.drop_columns_, X.columns)
        else:
            drop_columns = self.drop_columns_

        if drop_columns is None:
            return X
        else:
            Xt = X.drop(drop_columns, axis=1)
            return Xt


class DropHighCorrelation(BaseEstimator, TransformerMixin):
    """
    Delete features that are highly correlated to each other.\
    Best correlated feature against target variable will be\
    selected from the highly correlated pairs within X.

    :param float threshold: Threshold value for Pearson's correlation \
    coefficient. (default=0.95)
    """

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fit transformer by identifying highly correlated variable pairs\
        and dropping one that is less correlated to the target variable.\
        Missing values will be imputed by mean.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X_y(X, y)

        self.drop_columns_ = None

        Xm = X.corr()
        pairs = []
        for feature in Xm.columns:
            pair = Xm[feature][abs(Xm[feature]) >= self.threshold].index.tolist()
            if len(pair) > 1:
                pairs.append(pair)

        unique_pairs = pd.DataFrame(pairs).drop_duplicates().to_numpy()

        for pair in unique_pairs:
            pearsons = []

            for col in pair:
                if col is not None:
                    pearson = np.corrcoef(X[col].fillna(X[col].mean()), y)[0][1]
                    pearsons.append(abs(pearson))
            best_col = pair[pearsons.index(max(pearsons))]

            for col in pair:
                if col != best_col and col is not None:
                    if self.drop_columns_ is None:
                        self.drop_columns_ = col
                    else:
                        self.drop_columns_ = np.append(self.drop_columns_, col)

        self.drop_columns_ = np.unique(self.drop_columns_)

        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        if self.drop_columns_[0] is not None:
            drop_columns = np.intersect1d(self.drop_columns_, X.columns)
        else:
            drop_columns = self.drop_columns_[0]

        if drop_columns is None:
            return X
        else:
            Xt = X.drop(drop_columns, axis=1)
            return Xt


class ImputeNaN(BaseEstimator, TransformerMixin):
    """
    Look for NaN values in the dataframe and impute by\
    strategy such as mean, median and mode.

    :param string cat_strategy: Strategy for imputing NaN exist in categorical\
    columns. If any other string apart from mode is specified, the\
    NaN will be imputed by fixed string name ImputedNaN. (default='mode')
    :param string num_strategy: Strategy for imputing NaN exist in numerical\
    columns. Either mean, median or mode can be specified and if any\
    other string is specified, mean imputation will be employed. (default='mean')
    """

    def __init__(self, cat_strategy='mode', num_strategy='mean'):
        self.cat_strategy = cat_strategy
        self.num_strategy = num_strategy
    
    def fit(self, X, y=None):
        """
        Fit transformer by identifying numerical and categorical\
        columns. Then, based on the strategy fit will store the\
        values used for NaN existing in each columns.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)
        
        self.num_columns_ = X.select_dtypes('number').columns
        self.cat_columns_ = X.select_dtypes(exclude='number').columns

        self.num_imputes_ = {}
        self.cat_imputes_ = {}

        for col in self.num_columns_:
            if self.num_strategy == 'mean':
                self.num_imputes_[col] = X[col].mean()
            elif self.num_strategy == 'median':
                self.num_imputes_[col] = X[col].median()
            elif self.num_strategy == 'mode':
                self.num_imputes_[col] = X[col].mode()[0]
            else:
                self.num_imputes_[col] = X[col].mean()
        
        for col in self.cat_columns_:
            if self.cat_strategy == 'mode':
                self.cat_imputes_[col] = X[col].mode()[0]
            else:
                self.cat_imputes_[col] = 'ImputedNaN'
        
        return self

    def transform(self, X):
        """
        Transform X by imputing with values obtained from\
        fitting stage.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        num_columns = np.intersect1d(self.num_columns_, X.columns)
        cat_columns = np.intersect1d(self.cat_columns_, X.columns)

        for col in num_columns:
            Xt[col] = Xt[col].fillna(self.num_imputes_[col])

        for col in cat_columns:
            Xt[col] = Xt[col].fillna(self.cat_imputes_[col])

        return Xt


class OneHotEncoding(BaseEstimator, TransformerMixin):
    """
    One Hot Encoding of categorical variables.

    :param bool drop_first: Whether to drop first column after one \
    hot encoding in order to avoid multi-collinearity. (default=True)
    """
    def __init__(self, drop_first=True):
        self.drop_first = drop_first

    def fit(self, X, y=None):
        """
        Fit transformer by getting column names after\
        one hot encoding.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.dummy_cols_ = pd.get_dummies(X, drop_first=self.drop_first).columns
        self.cat_columns_ = X.select_dtypes(exclude='number').columns
        return self

    def transform(self, X):
        """
        Transform X by one hot encoding. This will drop new columns\
        in the input and impute with dummy column with all values = 0\
        for column that has been deleted from fitting stage.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = pd.get_dummies(X, drop_first=self.drop_first)
        self.new_dummy_cols_ = Xt.columns

        for col in np.setdiff1d(self.dummy_cols_, self.new_dummy_cols_):
            Xt[col] = 0
        for col in np.setdiff1d(self.new_dummy_cols_, self.dummy_cols_):
            Xt = Xt.drop(col, axis=1)

        Xt = pd.DataFrame(Xt, columns=self.dummy_cols_)

        return Xt


class BinarizeNaN(BaseEstimator, TransformerMixin):
    """
    Find a column with missing values, and create a new\
    column indicating whether a value was missing (0) or\
    not (1).
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit transformer by getting column names that\
        contains NaN

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        nan_info = X.isna().sum()
        self.nan_columns_ = nan_info[nan_info != 0].index
        return self

    def transform(self, X):
        """Transform by checking for columns containing NaN value\
        both during the fitting and transforming stage, then\
        binalizing NaN to a new column.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        new_nan_info = Xt.isna().sum()
        new_nan_columns = new_nan_info[new_nan_info != 0].index

        for col in np.intersect1d(self.nan_columns_, new_nan_columns):
            Xt[col + '_NaNFlag'] = Xt[col].isna().apply(lambda x: 1 if x else 0)
        for col in np.setdiff1d(self.nan_columns_, new_nan_columns):
            Xt[col + '_NaNFlag'] = 0

        return Xt


class CountRowNaN(BaseEstimator, TransformerMixin):
    """
    Calculates total number of NaN in a row and create
    a new column to store the total.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit transformer by getting column names during fit.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.cols_ = X.columns
        return self

    def transform(self, X):
        """
        Transform by checking for columns that exists in both\
        the fitting and transforming stage. Then adds up number of\
        missing values in row direction.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        check_columns = np.intersect1d(self.cols_, Xt.columns)

        Xt['NaN_Totals'] = Xt[check_columns].isna().sum(axis=1)
        return Xt


class StandardizeData(BaseEstimator, TransformerMixin):
    """
    Standardize datasets to have mean = 0 and std = 1.\
    Note this will only standardize numerical data\
    and ignore missing values during computation.\
    Deprecated in version 1.1.0 and will be removed in \
    version 1.3.0. Please use StandardScaling instead.
    """

    def __init__(self):
        warnings.warn("Deprecated in version 1.1.0 and will be\
 removed in version 1.3.0. Please use StandardScaling instead."
, FutureWarning)

    def fit(self, X, y=None):
        """
        Fit transformer to get mean and std for each\
        numerical features.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.num_columns_ = X.select_dtypes('number').columns
        
        self.dic_mean_ = {}
        self.dic_std_ = {}
        for col in self.num_columns_:
            self.dic_mean_[col] = X[col].mean()
            self.dic_std_[col] = X[col].std()
        return self

    def transform(self, X):
        """
        Transform by subtracting mean and dividing by std.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        standardize_columns = np.intersect1d(self.num_columns_, Xt.columns)
        for col in standardize_columns:
            if self.dic_std_[col] == 0:
                pass
            else:
                Xt[col] = (Xt[col] - self.dic_mean_[col]) / self.dic_std_[col]
        
        return Xt


class ClipData(BaseEstimator, TransformerMixin):
    """
    Clip datasets by replacing values larger than\
    the upper bound with upper bound and lower than \
    the lower bound by lower bound. Missing values will\
    be ignored.

    :param float threshold: Threshold value for to define upper and \
    lower bound. For example, 0.99 will imply upper bound at 99% percentile\
    annd lower bound at 1% percentile. (default=0.99)
    """
    def __init__(self, threshold=0.99):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fit transformer to get upper bound and lower bound for\
        numerical columns.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.num_columns_ = X.select_dtypes('number').columns
        self.upperbounds_ = {}
        self.lowerbounds_ = {}
        for col in self.num_columns_:
            self.upperbounds_[col], self.lowerbounds_[col] = np.percentile(
                X[col].dropna(), [100-self.threshold*100, self.threshold*100])

        return self

    def transform(self, X):
        """
        Transform by clipping numerical data

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        clip_columns = np.intersect1d(self.num_columns_, Xt.columns)
        for col in clip_columns:
            Xt[col] = np.clip(Xt[col].copy(), 
                              self.upperbounds_[col],
                              self.lowerbounds_[col])

        return Xt


class GroupRareCategory(BaseEstimator, TransformerMixin):
    """
    Replace rare categories that appear in categorical columns\
    with dummy string.

    :param float threshold: Threshold value for defining "rare"\
    category. For example, 0.01 will imply 1% of the total number\
    of data as "rare". (default=0.01)
    """
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fit transformer to define and store rare categories\
        to be replaced.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.cat_columns_ = X.select_dtypes(exclude='number').columns

        self.rare_categories_ = {}
        for col in self.cat_columns_:
            catcounts = X[col].value_counts(ascending=False)
            rare_categories = catcounts[catcounts <=
                    catcounts.sum() * self.threshold].index.tolist()
            self.rare_categories_[col] = rare_categories
        return self

    def transform(self, X):
        """
        Transform by replacing rare category with dummy string.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        group_columns = np.intersect1d(self.cat_columns_, Xt.columns)
        for col in group_columns:
            rare_categories = self.rare_categories_[col]

            for cat in rare_categories:
                Xt[col] = Xt[col].replace(cat, 'RareCategory')

        return Xt


class TargetMeanEncoding(BaseEstimator, TransformerMixin):
    """
    Target Mean Encoding of categorical variables. Missing\
    values will be treated as one of the categories.

    :param float k: hyperparameter for sigmoid function (default=0.0)
    :param float f: hyperparameter for sigmoid function (default=1.0)
    :param float smoothing: Whether to smooth target mean with global mean using\
    sigmoid function. Do not recommend smoothing=False. (default=0.01)
    """
    def __init__(self, k=0, f=1, smoothing=True):
        self.k = k
        self.f = f
        self.smoothing = smoothing
    
    def _sigmoid(self, count, k, f):
        return 1 / (1 + np.exp(- (count - k) / f))

    def fit(self, X, y=None):
        """
        Fit transformer to define and store target mean \
        smoothed target mean for categorical variables.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Input Series for target variable
        :return: fitted object (self)
        :rtype: object
        """
        _check_X_y(X, y)

        target = y.name
        global_mean = y.mean()
        self.global_mean_ = global_mean
        sigmoid = np.vectorize(self._sigmoid)
        self.cat_columns_ = X.select_dtypes(exclude='number').columns

        self.dic_target_mean_ = {}
        for col in self.cat_columns_:
            df = pd.concat([X[col], y], axis=1).fillna('_Missing'
                    ).groupby(col, as_index=False)
            local_means = df.mean().rename(columns={target:'target_mean'})
            counts = df.count().rename(columns={target:'count'})

            df_summary = pd.merge(counts, local_means, on=col)
            lambda_ = sigmoid(df_summary['count'], self.k, self.f)

            df_summary['smoothed_target_mean'] = lambda_ * df_summary[
                    'target_mean'] + (1 - lambda_) * global_mean
            df_summary.loc[df_summary['count'] == 1,
                    'smoothed_target_mean'] = global_mean
            self.dic_target_mean_[col] = df_summary

        return self

    def transform(self, X):
        """
        Transform by replacing categories with smoothed target mean

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        encode_columns = np.intersect1d(self.cat_columns_, Xt.columns)
        
        for col in encode_columns:
            df_map = self.dic_target_mean_[col][[col,
                    'smoothed_target_mean']].fillna('_Missing').set_index(col)
            Xt[col] = Xt[[col]].fillna('_Missing').join(df_map, on=col).drop(col, axis=1)

            Xt[col] = Xt[col].fillna(self.global_mean_)

        return Xt


class StandardScaling(BaseEstimator, TransformerMixin):
    """
    Standardize datasets to have mean = 0 and std = 1.\
    Note this will only standardize numerical data\
    and ignore missing values during computation.\
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit transformer to get mean and std for each\
        numerical features.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.num_columns_ = X.select_dtypes('number').columns
        
        self.dic_mean_ = {}
        self.dic_std_ = {}
        for col in self.num_columns_:
            self.dic_mean_[col] = X[col].mean()
            self.dic_std_[col] = X[col].std()
        return self

    def transform(self, X):
        """
        Transform by subtracting mean and dividing by std.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        standardize_columns = np.intersect1d(self.num_columns_, Xt.columns)
        for col in standardize_columns:
            if self.dic_std_[col] == 0:
                pass
            else:
                Xt[col] = (Xt[col] - self.dic_mean_[col]) / self.dic_std_[col]
        
        return Xt


class MinMaxScaling(BaseEstimator, TransformerMixin):
    """
    Rescale the fit data into range between 0 and 1.\
    Note this will only standardize numerical data\
    and ignore missing values during computation.\
    If there are values larger/smaller than fit data in the\
    transform data, the value will be larger than 1\
    or less than 0.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit transformer to get min and max values for each\
        numerical features.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.num_columns_ = X.select_dtypes('number').columns
        
        self.dic_min_ = {}
        self.dic_max_ = {}
        for col in self.num_columns_:
            self.dic_min_[col] = X[col].min()
            self.dic_max_[col] = X[col].max()
        return self

    def transform(self, X):
        """
        Transform by subtracting min and dividing by max-min.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        minmax_columns = np.intersect1d(self.num_columns_, Xt.columns)
        for col in minmax_columns:
            if self.dic_max_[col] == self.dic_min_[col]:
                pass
            else:
                Xt[col] = (Xt[col] - self.dic_min_[col]) / (
                        self.dic_max_[col] - self.dic_min_[col])
        
        return Xt


class CountEncoding(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables by the count of category\
    within the categorical column.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit transformer to define categorical variables and\
        obtain occurrence of each categories.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.cat_columns_ = X.select_dtypes(exclude='number').columns

        self.dic_counts_ = {}
        for col in self.cat_columns_:
            df = pd.concat([X[col], pd.DataFrame(np.zeros(X.shape[0]))],
                    axis=1).fillna('_Missing').groupby(col, as_index=False)
            counts = df.count().rename(columns={0:'count'})
            self.dic_counts_[col] = counts

        return self

    def transform(self, X):
        """
        Transform by replacing categories with counts

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        encode_columns = np.intersect1d(self.cat_columns_, Xt.columns)
        
        for col in encode_columns:
            df_map = self.dic_counts_[col].fillna('_Missing').set_index(col)
            Xt[col] = Xt[[col]].fillna('_Missing').join(df_map, on=col).drop(col, axis=1)

            Xt[col] = Xt[col].fillna(0)

        return Xt


class RankedCountEncoding(BaseEstimator, TransformerMixin):
    """
    Firstly encode categorical variables by the count of category\
    within the categorical column. Then, counts are ranked in\
    descending order and the ranks are used to encode category\
    columns. Even in case there are categories with same counts,\
    ranking will be based on the index and therefore the\
    categories will be distinguished. RankedFrequencyEncoding\
    is not provided as the result will be identical to this class.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit transformer to define categorical variables and\
        obtain ranking of category counts.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.cat_columns_ = X.select_dtypes(exclude='number').columns

        self.dic_ranks_ = {}
        for col in self.cat_columns_:
            df_rank = pd.DataFrame(X[col].fillna('_Missing'
                    ).value_counts(ascending=False)).reset_index().reset_index()
            df_rank.columns = ['Rank', col, 'Counts']
            df_rank['Rank'] += 1
            df_rank = df_rank.set_index(col)
            self.dic_ranks_[col] = df_rank

        return self

    def transform(self, X):
        """
        Transform by replacing categories with ranks

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        encode_columns = np.intersect1d(self.cat_columns_, Xt.columns)
        
        for col in encode_columns:
            df_rank = self.dic_ranks_[col]
            Xt[col] = Xt[[col]].fillna('_Missing').join(df_rank,
                    on=col).drop([col, 'Counts'], axis=1)

            Xt[col] = Xt[col].fillna(df_rank['Rank'].max() + 1)

        return Xt


class FrequencyEncoding(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables by the frequency of category\
    within the categorical column.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit transformer to define categorical variables and\
        obtain frequency of each categories.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.cat_columns_ = X.select_dtypes(exclude='number').columns

        self.dic_freq_ = {}
        for col in self.cat_columns_:
            df = pd.concat([X[col], pd.DataFrame(np.zeros(X.shape[0]))],
                    axis=1).fillna('_Missing').groupby(col, as_index=False)
            df_count = df.count()
            df_count.columns = [col, 'Frequency']
            df_count['Frequency'] = df_count[['Frequency']].apply(
                    lambda x: x / x.sum())
            self.dic_freq_[col] = df_count

        return self

    def transform(self, X):
        """
        Transform by replacing categories with frequency

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        encode_columns = np.intersect1d(self.cat_columns_, Xt.columns)
        
        for col in encode_columns:
            df_map = self.dic_freq_[col].fillna('_Missing').set_index(col)
            Xt[col] = Xt[[col]].fillna('_Missing').join(df_map, on=col).drop(col, axis=1)

            Xt[col] = Xt[col].fillna(0)

        return Xt


class RankedTargetMeanEncoding(BaseEstimator, TransformerMixin):
    """
    Ranking with Target Mean Encoding of categorical variables. Missing\
    values will be treated as one of the categories. This will treat\
    Categories with same target mean separately as the rank is obtained\
    from index once sorted by target mean.

    :param float k: hyperparameter for sigmoid function (default=0.0)
    :param float f: hyperparameter for sigmoid function (default=1.0)
    :param float smoothing: Whether to smooth target mean with global mean using\
    sigmoid function. Do not recommend smoothing=False. (default=0.01)
    """
    def __init__(self, k=0, f=1, smoothing=True):
        self.k = k
        self.f = f
        self.smoothing = smoothing
    
    def _sigmoid(self, count, k, f):
        return 1 / (1 + np.exp(- (count - k) / f))

    def fit(self, X, y=None):
        """
        Fit transformer to define and store target mean \
        smoothed target mean for categorical variables.\
        Then, ranking is created based on target mean.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Input Series for target variable
        :return: fitted object (self)
        :rtype: object
        """
        _check_X_y(X, y)

        target = y.name
        global_mean = y.mean()
        self.global_mean_ = global_mean
        sigmoid = np.vectorize(self._sigmoid)
        self.cat_columns_ = X.select_dtypes(exclude='number').columns

        self.dic_target_mean_ = {}
        for col in self.cat_columns_:
            df = pd.concat([X[col], y], axis=1).fillna('_Missing'
                    ).groupby(col, as_index=False)
            local_means = df.mean().rename(columns={target:'target_mean'})
            counts = df.count().rename(columns={target:'count'})

            df_summary = pd.merge(counts, local_means, on=col)
            lambda_ = sigmoid(df_summary['count'], self.k, self.f)

            df_summary['smoothed_target_mean'] = lambda_ * df_summary[
                    'target_mean'] + (1 - lambda_) * global_mean
            df_summary.loc[df_summary['count'] == 1,
                    'smoothed_target_mean'] = global_mean
            
            df_summary = df_summary.sort_values('smoothed_target_mean'
                    , ascending=False).reset_index(drop=True).reset_index()
            df_summary = df_summary.rename(columns={'index':'Rank'})
            df_summary['Rank'] += 1

            self.dic_target_mean_[col] = df_summary

        return self

    def transform(self, X):
        """
        Transform by replacing categories with rank

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        encode_columns = np.intersect1d(self.cat_columns_, Xt.columns)
        
        for col in encode_columns:
            df_map = self.dic_target_mean_[col][[col,
                    'Rank']].fillna('_Missing').set_index(col)
            Xt[col] = Xt[[col]].fillna('_Missing').join(df_map, on=col).drop(col, axis=1)

            Xt[col] = Xt[col].fillna(df_map['Rank'].max() + 1)

        return Xt


class AppendAnomalyScore(BaseEstimator, TransformerMixin):
    """
    Append anomaly score calculated from isolation forest.\
    Since IsolationForest needs to be fitted, category columns must\
    first be encoded to numerical values.

    :param int n_estimators: Number of base estimators in the \
        Isolation Forest ensemble. (default=100)
    :param int random_state: random_state for Isolation Forest \
        (default=1234)
    """

    def __init__(self, n_estimators=100, random_state=1234):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit Isolation Forest

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.model_ = IsolationForest(n_estimators=self.n_estimators,
                                      random_state=self.random_state)
        self.model_.fit(X)
        self.fit_columns_ = X.columns

        return self

    def transform(self, X):
        """
        Transform X by appending anomaly score

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        _check_fit(self.fit_columns_, Xt.columns)
        Xt['Anomaly_Score'] = list(self.model_.decision_function(Xt))
        return Xt


class AppendCluster(BaseEstimator, TransformerMixin):
    """
    Append cluster number obtained from kmeans++ clustering.\
    For clustering categorical variables need to be converted\
    to numerical data.

    :param int n_clusters: Number of clusters (default=8)
    :param int random_state: random_state for KMeans \
        (default=1234)
    """

    def __init__(self, n_clusters=8, random_state=1234):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit KMeans Clustering

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.model_ = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state)
        self.model_.fit(X)
        self.fit_columns_ = X.columns

        return self

    def transform(self, X):
        """
        Transform X by appending cluster number

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        _check_fit(self.fit_columns_, Xt.columns)
        Xt['Cluster_Number'] = list(self.model_.predict(Xt))
        return Xt


class AppendClusterDistance(BaseEstimator, TransformerMixin):
    """
    Append cluster distance obtained from kmeans++ clustering.\
    For clustering categorical variables need to be converted\
    to numerical data.

    :param int n_clusters: Number of clusters (default=8)
    :param int random_state: random_state for KMeans \
        (default=1234)
    """

    def __init__(self, n_clusters=8, random_state=1234):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit KMeans Clustering

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.model_ = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state)
        self.model_.fit(X)
        self.fit_columns_ = X.columns

        return self

    def transform(self, X):
        """
        Transform X by appending cluster distance

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        _check_fit(self.fit_columns_, Xt.columns)
        df_clusters = pd.DataFrame(self.model_.transform(Xt)
                 ).add_prefix('Cluster_Distance_')
        df_clusters.index = Xt.index

        Xt = pd.concat([Xt, df_clusters], axis=1)

        return Xt


class AppendPrincipalComponent(BaseEstimator, TransformerMixin):
    """
    Append principal components obtained from PCA.\
    For pca categorical variables need to be converted\
    to numerical data. Also, data should be standardized beforehand.

    :param int n_components: Number of principal components (default=5)
    :param int random_state: random_state for PCA \
        (default=1234)
    """

    def __init__(self, n_components=5, random_state=1234):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit PCA

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)

        self.model_ = PCA(n_components=self.n_components,
                          random_state=self.random_state)
        self.model_.fit(X)
        self.fit_columns_ = X.columns

        return self

    def transform(self, X):
        """
        Transform X by appending principal components

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        _check_fit(self.fit_columns_, Xt.columns)
        df_pca = pd.DataFrame(self.model_.transform(Xt)
                 ).add_prefix('Principal_Component_')
        df_pca.index = Xt.index

        Xt = pd.concat([Xt, df_pca], axis=1)

        return Xt


class ArithmeticFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer which recognizes all numerical features and create\
    new features by arithmetic operation. Newly created features\
    are evaluated individually by fitting Logistic Regression against\
    the target variable and only new features with higher eval metric than feature\
    pairs will be newly added to the data. Missing values need to be\
    imputed beforehand.

    :param int max_features: Number of numerical features to test\
    combinations. If number of numerical features in the data exceeds\
    this value, transformer will raise an exception. (default=50)
    :param string metric: Metrics to evaluate feature. Sklearn default\
    metrics can be used. (default='roc_auc')
    :param string operation: Type of arithmetic operations. 'add', \
    'subtract', 'multiply', 'divide' can be used. (default='multiply')
    :param float replace_zero: Value to replace 0 when operation='divide'\
    . Do not use 0 as it may cause ZeroDivisionError.(default=0.001)
    """

    def __init__(self,
                 max_features=50,
                 metric='roc_auc',
                 operation='multiply',
                 replace_zero=0.001):
        self.max_features = max_features
        self.metric = metric
        self.operation = operation
        self.replace_zero = replace_zero
    
    def _arithmetic_operation(self, series1, series2, operation):
        if operation == 'add':
            return pd.DataFrame(series1 + series2)
        elif operation == 'subtract':
            return pd.DataFrame(series1 - series2)
        elif operation == 'multiply':
            return pd.DataFrame(series1 * series2)
        elif operation == 'divide':
            return pd.DataFrame(np.divide(series1, series2.replace(0, self.replace_zero)))
        else:
            raise Exception('Unknown arithmetic operation was specified : ' + operation)
    
    def _check_missing(self, df):
        if df.isna().sum().sum() != 0:
            raise Exception('Please impute missing values before using this transformer.')

    def fit(self, X, y=None):
        """
        Fit transformer by fitting each feature with Logistic\
        Regression and storing features with eval metrics higher\
        than the max of existing features.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Input Series for target variable
        :return: fitted object (self)
        :rtype: object
        """
        _check_X_y(X, y)
        _check_binary(y)

        self.x_features_ = X.select_dtypes('number').columns
        self._check_missing(X[self.x_features_])
        
        if len(self.x_features_) > self.max_features:
            raise Exception('Number of numerical features is larger than max_features.')

        cv = StratifiedKFold(n_splits=5)
        lr = LogisticRegression(penalty='l2', solver='lbfgs')
        
        # Firstly find maximum evaluation metric in the existing feature
        roc_auc_existing = {}
        for feature in self.x_features_:
            X_lr = X[[feature]].fillna(X[[feature]].mean())
        
            roc_auc = cross_val_score(lr, X_lr, y, cv=cv, scoring=self.metric).mean()
            roc_auc_existing[feature] = roc_auc

        # Create feature by multiplication and employ if evaluation metric
        # is larger than the maximum of existing features
        combinations = list(itertools.combinations(self.x_features_, 2))
        self.new_pair_ = []
        for pair in combinations:
            X_lr = self._arithmetic_operation(X[pair[0]], X[pair[1]], self.operation)

            roc_auc = cross_val_score(lr, X_lr, y, cv=cv, scoring=self.metric).mean()
            max_auc_pair = max(roc_auc_existing[pair[0]],
                               roc_auc_existing[pair[1]])

            if roc_auc > max_auc_pair:
                self.new_pair_.append(pair)

        return self

    def transform(self, X):
        """
        Transform X by creating new feature using pairs identified during fit.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)
        self._check_missing(X[self.x_features_])

        Xt = X.copy()

        if not self.new_pair_:
            return X
        else:
            for pair in self.new_pair_:
                Xt[pair[0] + '_' + self.operation + '_' + pair[1]] = self._arithmetic_operation(
                        Xt[pair[0]], Xt[pair[1]], self.operation)
            return Xt


class RankedEvaluationMetricEncoding(BaseEstimator, TransformerMixin):
    """
    Encode categorical columns by firstly creating dummy variable, then\
    LogisticRegression against target variable is fitted\
    for each of the dummy variables. Evaluation metric such as accuracy\
    or roc_auc is calculated and ranked. Finally, categories are encoded\
    with its rank. It is strongly recommended to conduct DropHighCardinality\
    or GroupRareCategory before using this encoding as this encoder will\
    fit Logistic Regression for ALL categories with 5-fold.

    :param string metric: Metrics to evaluate feature. Sklearn default\
    metrics can be used. (default='roc_auc')
    """

    def __init__(self, metric='roc_auc'):
        self.metric = metric

    def fit(self, X, y=None):
        """
        Fit transformer by creating dummy variable and fitting \
        LogisticRegression.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Input Series for target variable
        :return: fitted object (self)
        :rtype: object
        """
        _check_X_y(X, y)
        _check_binary(y)

        self.cat_columns_ = X.select_dtypes(exclude='number').columns
        
        cv = StratifiedKFold(n_splits=5)
        lr = LogisticRegression(penalty='l2', solver='lbfgs')

        self.dic_corr_ = {}
        for feature in self.cat_columns_:
            X_lr = pd.get_dummies(X[[feature]].fillna('_Missing'))
            df_map = pd.DataFrame([])
            for col in X_lr.columns:
                eval_metric = cross_val_score(lr, X_lr[[col]], y, cv=cv, scoring=self.metric).mean()
                df_map = pd.concat([df_map, pd.DataFrame([col.replace(feature + '_', ''),
                        eval_metric]).T], axis=0)
            df_map.columns = ['Category', 'Evaluation_Metric']

            df_map = df_map.sort_values('Evaluation_Metric'
                    , ascending=False).reset_index(drop=True).reset_index()
            df_map = df_map.rename(columns={'index':'Rank'})
            df_map['Rank'] += 1
            df_map = df_map.set_index('Category')

            self.dic_corr_[feature] = df_map

        return self

    def transform(self, X):
        """
        Transform X by replacing categories with its evaluation metric

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        
        encode_columns = np.intersect1d(self.cat_columns_, Xt.columns)
        
        for col in encode_columns:
            df_map = self.dic_corr_[col]
            df_map = df_map[['Rank']]
            Xt[col] = Xt[[col]].fillna('_Missing').join(df_map, on=col).drop(col, axis=1)
            Xt[col] = Xt[col].fillna(0)

        return Xt


class AppendClassificationModel(BaseEstimator, TransformerMixin):
    """
    Append prediction from model as a new feature. Model must have\
    fit and predict methods and it should only predict a single\
    label. In case the model has a predict_proba method, option\
    probability can be used to append class probability instead\
    of class labels. predict_proba method must return class\
    probability for 0 as first column and 1 as second column.

    :param object model: Any model that is in line with sklearn \
    classification model, meaning it implements fit and predict.\
    (default=None)
    :param bool probability: Whether to class probability instead \
    of class labels. If True, model must have predict_proba method\
    implemented.(default=False)
    """

    def __init__(self, model=None, probability=False):
        self.model = model
        self.probability = probability

    def fit(self, X, y=None):
        """
        Fit transformer by fitting model specified.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Input Series for target variable
        :return: fitted object (self)
        :rtype: object
        """
        _check_X_y(X, y)
        _check_binary(y)

        self.model_ = self.model
        _check_method_implemented(self.model_, 'fit')
        _check_method_implemented(self.model_, 'predict')
        
        if self.probability:
            _check_method_implemented(self.model_, 'predict_proba')

        self.model_.fit(X, y)
        self.fit_columns_ = X.columns

        return self

    def transform(self, X):
        """
        Transform X by predicting with fitted model.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        _check_fit(self.fit_columns_, Xt.columns)
        
        if self.probability:
            try:
                y_pred = self.model_.predict_proba(Xt)[:, 1]
            except:
                y_pred = self.model_.predict_proba(Xt)
        else:
            y_pred = self.model_.predict(Xt)
        
        pred_name = 'Predicted_' + type(self.model_).__name__

        Xt[pred_name] = y_pred

        return Xt


class AppendEncoder(BaseEstimator, TransformerMixin):
    """
    Append encoders in the DataLiner module. Encoders in DataLiner\
    will automatically replace categorical values, but by wrapping\
    DataLiner Encoders with this class, encoded results will be\
    appended as a new feature and original categorical columns\
    will remain. Regardless of whether the Encoder will require\
    target column or not, this class will require target column.

    :param object encoder: DataLiner Encoders.(default=None)
    """

    def __init__(self, encoder=None):
        self.encoder = encoder

    def fit(self, X, y=None):
        """
        Fit transformer by fitting encoder specified

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Input Series for target variable
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)
        if y is not None:
            _check_X_y(X, y)

        self.encoder_ = self.encoder

        self.encoder_.fit(X, y)
        self.fit_columns_ = X.columns

        return self

    def transform(self, X):
        """
        Transform X by appending encoded category

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        _check_fit(self.fit_columns_, Xt.columns)

        name = '_' + type(self.encoder_).__name__

        if name == '_OneHotEncoding':
            Xa = self.encoder_.transform(X[self.encoder_.cat_columns_]
                    ).add_suffix(name)
        else:
            Xa = self.encoder_.transform(X
                    )[self.encoder_.cat_columns_].add_suffix(name)

        Xt = pd.concat([Xt, Xa], axis=1)

        return Xt


class AppendClusterTargetMean(BaseEstimator, TransformerMixin):
    """
    Append cluster number obtained from kmeans++ clustering.\
    Then each cluster number is replaced with target mean.\
    For clustering categorical variables need to be converted\
    to numerical data.

    :param int n_clusters: Number of clusters (default=8)
    :param int random_state: random_state for KMeans \
        (default=1234)
    """

    def __init__(self, n_clusters=8, random_state=1234):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _sigmoid(self, count, k=0, f=1):
        return 1 / (1 + np.exp(- (count - k) / f))

    def fit(self, X, y=None):
        """
        Fit KMeans Clustering and obtain target mean

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X(X)
        _check_X_y(X, y)

        global_mean = y.mean()
        self.global_mean_ = global_mean

        self.model_ = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state)
        self.model_.fit(X)
        self.fit_columns_ = X.columns
        
        cluster = self.model_.predict(X)
        df_cluster = pd.concat([pd.DataFrame(cluster),
                                  pd.DataFrame(y)], axis=1)
        df_cluster.columns = ['Cluster_Number', 'Target']

        mean = df_cluster.groupby('Cluster_Number').mean().rename(
            columns={'Target':'target_mean'})
        count = df_cluster.groupby('Cluster_Number').count().rename(
            columns={'Target':'count'})
        df_map = pd.concat([mean, count], axis=1)
        lambda_ = self._sigmoid(df_map['count'])
        df_map['smoothed_target_mean'] = lambda_ * df_map[
                    'target_mean'] + (1 - lambda_) * global_mean
        df_map.loc[df_map['count'] == 1,
                'smoothed_target_mean'] = global_mean
        self.cluster_target_mean_ = df_map

        return self

    def transform(self, X):
        """
        Transform X by appending cluster mean

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        Xt = X.copy()
        _check_fit(self.fit_columns_, Xt.columns)
        Xc = pd.DataFrame(self.model_.predict(Xt)).rename(columns={0:'Cluster_Number'})
        
        df_map = self.cluster_target_mean_
        df_map = df_map[['smoothed_target_mean']]
        Xc = Xc.join(df_map, on='Cluster_Number').drop('Cluster_Number', axis=1)
        Xc = Xc.fillna(0)

        Xt = pd.concat([Xt, Xc.rename(columns={'smoothed_target_mean':'cluster_mean'})], axis=1)

        return Xt


class PermutationImportanceTest(BaseEstimator, TransformerMixin):
    """
    Conduct permutation importance tests on features and drop features\
    that are not effective. Basically it will firstly fit entire data,\
    then randomly shuffle each feature's data and evaluate the metrics\
    for both cases. If shuffled case has no difference in the evaluation\
    then that means the feature is not effective in prediction.

    :param float threshold: Average difference in roc_auc between original\
    and shuffled dataset. Higher the value, more features will be dropped.\
    (default=0.0001)
    """

    def __init__(self, threshold=0.0001):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Conduct permutation importance test and store drop features.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_X_y(X, y)
        _check_binary(y)

        process = make_pipeline(ImputeNaN(), TargetMeanEncoding())
        cv = StratifiedKFold(n_splits=5)
        clf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=1234)

        Xt = process.fit_transform(X, y)
        metrics = np.array([])

        feature_metrics_dic = {}
        for feature in X.columns:
            feature_metrics_dic[feature] = np.array([])

        for train_idx, valid_idx in cv.split(Xt, y):
            X_train, X_valid, y_train, y_valid = \
                    Xt.iloc[train_idx], Xt.iloc[valid_idx], y.iloc[train_idx], y.iloc[valid_idx]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_valid)[:, 1]
            metrics = np.append(metrics, roc_auc_score(y_valid, y_pred))
            
            for feature in X.columns:
                X_valid2 = X_valid.copy()
                X_shuffled = X_valid2[feature].sample(frac=1, random_state=1234)
                X_shuffled.index = X_valid2.index
                X_valid2[feature] = X_shuffled.copy()
                y_pred_shuffled = clf.predict_proba(X_valid2)[:, 1]
                feature_metrics_dic[feature] = np.append(feature_metrics_dic[feature],
                                                        roc_auc_score(y_valid, y_pred_shuffled))

        base_metric = metrics.mean()
        for feature in X.columns:
            feature_metrics_dic[feature] = base_metric - feature_metrics_dic[feature].mean()

        self.drop_columns_ = []
        for key, value in feature_metrics_dic.items():
            if value <= self.threshold:
                self.drop_columns_.append(key)

        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)
        _check_X(X)

        if self.drop_columns_ is not None:
            drop_columns = np.intersect1d(self.drop_columns_, X.columns)
        else:
            drop_columns = self.drop_columns_

        if drop_columns is None:
            return X
        else:
            Xt = X.drop(drop_columns, axis=1)
            return Xt
