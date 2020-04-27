# -*- coding: utf-8 -*-
"""
A dataprocessing package for data preprocess and feature engineering.

This library contains preprocessing methods for data processing
and feature engineering used during data analysis and machine learning 
process.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression


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
]


def _check_x(X):
    if isinstance(X, pd.DataFrame):
        pass
    else:
        raise TypeError("Input X is not a pandas DataFrame.")


def _check_y(y):
    if isinstance(y, pd.Series):
        pass
    else:
        raise TypeError("Input y is not a pandas Series.")


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Simply delete columns specified from input dataframe.

    :param list drop_columns: List of feature names which will be droped \
        from input dataframe. (default=None)
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
        _check_x(X)
        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        if self.drop_columns is None:
            return X
        else:
            drop_columns = np.intersect1d(self.drop_columns, X.columns)
            Xt = X.drop(drop_columns, axis=1)
            return Xt


class DropNoVariance(BaseEstimator, TransformerMixin):
    """
    Delete columns which only have single unique value.
    """

    def __init__(self):
        self.drop_columns = None

    def fit(self, X, y=None):
        """
        Fit transformer by deleting column with single unique value.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_x(X)

        for feature in X.columns:
            if X[feature].unique().shape[0] == 1:
                if self.drop_columns is None:
                    self.drop_columns = [feature]
                else:
                    self.drop_columns = np.append(self.drop_columns, feature)

        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        if self.drop_columns is None:
            return X
        else:
            drop_columns = np.intersect1d(self.drop_columns, X.columns)
            Xt = X.drop(drop_columns, axis=1)
            return Xt


class DropHighCardinality(BaseEstimator, TransformerMixin):
    """
    Delete columns with high cardinality.
    Basically means dropping column with too many categories.

    :param int max_categories: Maximum number of categories to be permitted\
    in a column. If number of categories in a certain column exceeds this value,\
    that column will be deleted. (default=50)
    """

    def __init__(self, max_categories=50):
        self.max_categories = max_categories
        self.drop_columns = None

    def fit(self, X, y=None):
        """
        Fit transformer by deleting column with high cardinality.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_x(X)

        for feature in X.columns:
            if X[feature].unique().shape[0] >= self.max_categories:
                if self.drop_columns is None:
                    self.drop_columns = [feature]
                else:
                    self.drop_columns = np.append(self.drop_columns, feature)

        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        if self.drop_columns is None:
            return X
        else:
            drop_columns = np.intersect1d(self.drop_columns, X.columns)
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
        self.drop_columns = None

    def fit(self, X, y=None):
        """
        Fit transformer by fitting each feature with Logistic \
        Regression and storing features with roc_auc less than threshold 

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Input Series for target variable
        :return: fitted object (self)
        :rtype: object
        """
        _check_x(X)
        _check_y(y)

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
                if self.drop_columns is None:
                    self.drop_columns = [feature]
                else:
                    self.drop_columns = np.append(self.drop_columns, feature)

        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        if self.drop_columns is None:
            return X
        else:
            drop_columns = np.intersect1d(self.drop_columns, X.columns)
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
        self.drop_columns = None

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
        _check_x(X)
        _check_y(y)

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
                    if self.drop_columns is None:
                        self.drop_columns = col
                    else:
                        self.drop_columns = np.append(self.drop_columns, col)
        if self.drop_columns is not None:
            self.drop_columns = sorted(set(self.drop_columns))
        return self

    def transform(self, X):
        """
        Transform X by dropping columns specified in drop_columns

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        if self.drop_columns is None:
            return X
        else:
            drop_columns = np.intersect1d(self.drop_columns, X.columns)
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
        _check_x(X)
        
        self.num_columns = X.select_dtypes('number').columns
        self.cat_columns = X.select_dtypes(exclude='number').columns

        self.num_imputes = {}
        self.cat_imputes = {}
        for col in self.num_columns:
            if self.num_strategy == 'mean':
                self.num_imputes[col] = X[col].mean()
            elif self.num_strategy == 'median':
                self.num_imputes[col] = X[col].median()
            elif self.num_strategy == 'mode':
                self.num_imputes[col] = X[col].mode()[0]
            else:
                self.num_imputes[col] = X[col].mean()
        
        for col in self.cat_columns:
            if self.cat_strategy == 'mode':
                self.cat_imputes[col] = X[col].mode()[0]
            else:
                self.cat_imputes[col] = 'ImputedNaN'
        
        return self

    def transform(self, X):
        """
        Transform X by imputing with values obtained from\
        fitting stage.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        Xt = X.copy()
        num_columns = np.intersect1d(self.num_columns, X.columns)
        cat_columns = np.intersect1d(self.cat_columns, X.columns)

        for col in num_columns:
            Xt[col] = Xt[col].fillna(self.num_imputes[col])

        for col in cat_columns:
            Xt[col] = Xt[col].fillna(self.cat_imputes[col])

        return Xt


class OneHotEncoding(BaseEstimator, TransformerMixin):
    """
    One Hot Encoding of categorical variables.

    :param boolean drop_first: Whether to drop first column after one \
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
        _check_x(X)
        self.dummy_cols = pd.get_dummies(X, drop_first=self.drop_first).columns
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
        Xt = pd.get_dummies(X, drop_first=self.drop_first)
        self.new_dummy_cols = Xt.columns

        for col in np.setdiff1d(self.dummy_cols, self.new_dummy_cols):
            Xt[col] = 0
        for col in np.setdiff1d(self.new_dummy_cols, self.dummy_cols):
            Xt = Xt.drop(col, axis=1)

        return Xt


class BinarizeNaN(BaseEstimator, TransformerMixin):
    """
    Find a column with missing values, and create a new\
    column indicating whether a value was missing (0) or\
    not (1).
    """
    def __init__(self):
        self.nan_columns = None

    def fit(self, X, y=None):
        """
        Fit transformer by getting column names that\
        contains NaN

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_x(X)
        nan_info = X.isna().sum()
        self.nan_columns = nan_info[nan_info != 0].index
        return self

    def transform(self, X):
        """Transform by checking for columns containing NaN value\
        both during the fitting and transforming stage, then\
        binalizing NaN to a new column.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        Xt = X.copy()
        new_nan_info = Xt.isna().sum()
        new_nan_columns = new_nan_info[new_nan_info != 0].index

        binalize_columns = np.intersect1d(self.nan_columns, new_nan_columns)

        for col in binalize_columns:
            Xt[col + '_NaNFlag'] = Xt[col].isna().apply(lambda x: 1 if x else 0)

        return Xt


class CountRowNaN(BaseEstimator, TransformerMixin):
    """
    Calculates total number of NaN in a row and create
    a new column to store the total.
    """

    def __init__(self):
        self.nan_columns = None

    def fit(self, X, y=None):
        """
        Fit transformer by getting column names during fit.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_x(X)
        self.cols = X.columns
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
        Xt = X.copy()
        check_columns = np.intersect1d(self.cols, Xt.columns)

        Xt['NaN_Totals'] = Xt[check_columns].isna().sum(axis=1)
        return Xt


class StandardizeData(BaseEstimator, TransformerMixin):
    """
    Standardize datasets to have mean = 0 and std = 1.\
    Note this will only standardize numerical data\
    and ignore missing values during computation.
    """

    def __init__(self):
        self.num_columns = None

    def fit(self, X, y=None):
        """
        Fit transformer to get mean and std for each\
        numerical features.

        :param pandas.DataFrame X: Input dataframe
        :param pandas.Series y: Ignored. (default=None)
        :return: fitted object (self)
        :rtype: object
        """
        _check_x(X)
        self.num_columns = X.select_dtypes('number').columns
        
        self.dic_mean = {}
        self.dic_std = {}
        for col in self.num_columns:
            self.dic_mean[col] = X[col].mean()
            self.dic_std[col] = X[col].std()
        return self

    def transform(self, X):
        """
        Transform by subtracting mean and dividing by std.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        Xt = X.copy()
        
        standardize_columns = np.intersect1d(self.num_columns, Xt.columns)
        for col in standardize_columns:
            if self.dic_std[col] == 0:
                pass
            else:
                Xt[col] = (Xt[col] - self.dic_mean[col]) / self.dic_std[col]
        
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
        _check_x(X)
        self.num_columns = X.select_dtypes('number').columns
        self.upperbounds = {}
        self.lowerbounds = {}
        for col in self.num_columns:
            self.upperbounds[col], self.lowerbounds[col] = np.percentile(
                X[col].dropna(), [100-self.threshold*100, self.threshold*100])
        return self

    def transform(self, X):
        """
        Transform by clipping numerical data

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        Xt = X.copy()
        
        clip_columns = np.intersect1d(self.num_columns, Xt.columns)
        for col in clip_columns:
            Xt[col] = np.clip(Xt[col].copy(), 
                              self.upperbounds[col],
                              self.lowerbounds[col])

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
        _check_x(X)
        self.cat_columns = X.select_dtypes(exclude='number').columns

        self.rare_categories = {}
        for col in self.cat_columns:
            catcounts = X[col].value_counts(ascending=False)
            rare_categories = catcounts[catcounts <=
                    catcounts.sum() * self.threshold].index.tolist()
            self.rare_categories[col] = rare_categories
        return self

    def transform(self, X):
        """
        Transform by replacing rare category with dummy string.

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        Xt = X.copy()
        
        group_columns = np.intersect1d(self.cat_columns, Xt.columns)
        for col in group_columns:
            rare_categories = self.rare_categories[col]

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
        _check_x(X)
        _check_y(y)
        target = y.name
        global_mean = y.mean()
        self.global_mean = global_mean
        sigmoid = np.vectorize(self._sigmoid)
        self.cat_columns = X.select_dtypes(exclude='number').columns

        self.dic_target_mean = {}
        for col in self.cat_columns:
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
            self.dic_target_mean[col] = df_summary

        return self

    def transform(self, X):
        """
        Transform by replacing categories with smoothed target mean

        :param pandas.DataFrame X: Input dataframe
        :return: Transformed input DataFrame
        :rtype: pandas.DataFrame
        """
        Xt = X.copy()
        
        encode_columns = np.intersect1d(self.cat_columns, Xt.columns)
        
        for col in encode_columns:
            df_map = self.dic_target_mean[col][[col,
                    'smoothed_target_mean']].fillna('_Missing').set_index(col)
            Xt[col] = Xt[[col]].fillna('_Missing').join(df_map, on=col).drop(col, axis=1)

            Xt[col] = Xt[col].fillna(self.global_mean)

        return Xt
