## DataLiner - Data processing package for Python 
A dataprocessing package for data preprocess and feature engineering.<br>
Please feel free to send pull requests for bug fix, improvements or new preprocessing methods!

## Installation
```
! pip install dataliner
```

## Quick Start
Train data from Kaggle Titanic is used in this example. https://www.kaggle.com/c/titanic/data

```python
import pandas as pd
from sklearn.pipeline import make_pipeline
import dataliner as dl

df = pd.read_csv('train.csv')
target_col = 'Survived'
X = df.drop(target_col, axis=1)
y = df[target_col]

process = make_pipeline(
    dl.DropNoVariance(),
    dl.DropHighCardinality(),
    dl.BinarizeNaN(),
    dl.ImputeNaN(),
    dl.TargetMeanEncoding(),
    dl.DropHighCorrelation(),
    dl.StandardizeData(),
    dl.DropLowAUC(),
)

process.fit_transform(X, y)

```

## Documentation
https://shallowdf20.github.io/dataliner/preprocessing.html

## What to expect in next version (1.1):
- Specifying columns to apply each preprocess
- Accepting numpy data types (currently pandas DataFrame / Series only)
- Additional preprocess (Data Normalization, Frequency Encoding, etc.)
- Strictly keeping the order of column names after preprocess
