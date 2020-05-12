## DataLiner - Data processing package for Python 
A dataprocessing package for data preprocess and feature engineering.<br>
Please feel free to send pull requests for bug fix, improvements or new preprocessing methods!

## Installation
```
! pip install dataliner
```

## Documentation
https://shallowdf20.github.io/dataliner/preprocessing.html

## Quick Start
Train data from Kaggle Titanic is used in this example. https://www.kaggle.com/c/titanic/data

```python
import dataliner as dl
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

X, X_test, y = dl.load_titanic()

process = make_pipeline(
    dl.DropColumns(drop_columns=['PassengerId']),
    dl.DropNoVariance(),
    dl.GroupRareCategory(threshold=0.01),
    dl.DropHighCardinality(max_categories=50),
    dl.DropLowAUC(threshold=0.51),
    dl.ClipData(threshold=0.99),
    dl.BinarizeNaN(),
    dl.CountRowNaN(),
    dl.ImputeNaN(cat_strategy='mode', num_strategy='mean'),
    dl.AppendEncoder(dl.TargetMeanEncoding(k=0, f=1, smoothing=True)),
    dl.OneHotEncoding(drop_first=True),
#     dl.TargetMeanEncoding(k=0, f=1, smoothing=True),
#     dl.CountEncoding(),
#     dl.RankedCountEncoding(),
#     dl.FrequencyEncoding(),
#     dl.RankedTargetMeanEncoding(k=0, f=1, smoothing=True),
#     dl.RankedEvaluationMetricEncoding(metric='roc_auc'),
    dl.StandardScaling(),
#     dl.MinMaxScaling(),
    dl.UnionAppend([
        dl.AppendCluster(n_clusters=8, random_state=1234),
        dl.AppendAnomalyScore(n_estimators=100, random_state=1234),
        dl.AppendPrincipalComponent(n_components=5, random_state=1234),
        dl.AppendClusterTargetMean(n_clusters=8, random_state=1234),
        dl.AppendClusterDistance(n_clusters=8, random_state=1234),
        dl.AppendArithmeticFeatures(max_features=50, metric='roc_auc', operation='add', replace_zero=0.001),
        dl.AppendArithmeticFeatures(max_features=50, metric='roc_auc', operation='subtract', replace_zero=0.001),
        dl.AppendArithmeticFeatures(max_features=50, metric='roc_auc', operation='multiply', replace_zero=0.001),
        dl.AppendArithmeticFeatures(max_features=50, metric='roc_auc', operation='divide', replace_zero=0.001),
#        dl.AppendClassificationModel(model=RandomForestClassifier(), probability=False)
    ]),
#     dl.DropLowAUC(),
    dl.PermutationImportanceTest(threshold=0.0001),
    dl.DropHighCorrelation(threshold=0.95),
)

Xt = process.fit_transform(X, y)
Xt_test = process.transform(X)

```
