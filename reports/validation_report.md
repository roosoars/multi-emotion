# Relatorio de validacao

- Modelo vencedor: `hist_gradient_boosting_blendshape`
- Feature set: `geometry_blendshape`
- Accuracy (teste): `0.5884`
- Macro-F1 (teste): `0.5825`
- Threshold de confianca: `0.60`

## Cobertura de deteccao
### train
- Anger: 910/1896 detectadas, 986 descartadas
- Disgust: 866/1891 detectadas, 1025 descartadas
- Fear: 957/1922 detectadas, 965 descartadas
- Happiness: 1018/1922 detectadas, 904 descartadas
- Sadness: 986/1922 detectadas, 936 descartadas
- Surprise: 949/1922 detectadas, 973 descartadas

### val
- Anger: 108/237 detectadas, 129 descartadas
- Disgust: 118/236 detectadas, 118 descartadas
- Fear: 127/240 detectadas, 113 descartadas
- Happiness: 136/240 detectadas, 104 descartadas
- Sadness: 128/240 detectadas, 112 descartadas
- Surprise: 124/240 detectadas, 116 descartadas

### test
- Anger: 122/237 detectadas, 115 descartadas
- Disgust: 119/237 detectadas, 118 descartadas
- Fear: 115/241 detectadas, 126 descartadas
- Happiness: 133/241 detectadas, 108 descartadas
- Sadness: 121/241 detectadas, 120 descartadas
- Surprise: 114/241 detectadas, 127 descartadas

## Selecionador de modelos
- hist_gradient_boosting_blendshape: macro-F1=0.5811, accuracy=0.5843, latencia=0.5146 ms/amostra
- random_forest_blendshape: macro-F1=0.5597, accuracy=0.5655, latencia=0.3852 ms/amostra
- logistic_regression_pca: macro-F1=0.3191, accuracy=0.3347, latencia=0.0027 ms/amostra

## Classification report
```json
{
  "Anger": {
    "precision": 0.6,
    "recall": 0.5409836065573771,
    "f1-score": 0.5689655172413793,
    "support": 122.0
  },
  "Disgust": {
    "precision": 0.6274509803921569,
    "recall": 0.5378151260504201,
    "f1-score": 0.579185520361991,
    "support": 119.0
  },
  "Fear": {
    "precision": 0.5142857142857142,
    "recall": 0.46956521739130436,
    "f1-score": 0.4909090909090909,
    "support": 115.0
  },
  "Happiness": {
    "precision": 0.6666666666666666,
    "recall": 0.6766917293233082,
    "f1-score": 0.6716417910447762,
    "support": 133.0
  },
  "Sadness": {
    "precision": 0.5849056603773585,
    "recall": 0.768595041322314,
    "f1-score": 0.6642857142857143,
    "support": 121.0
  },
  "Surprise": {
    "precision": 0.5221238938053098,
    "recall": 0.5175438596491229,
    "f1-score": 0.5198237885462555,
    "support": 114.0
  },
  "accuracy": 0.5883977900552486,
  "macro avg": {
    "precision": 0.5859054859212011,
    "recall": 0.5851990967156411,
    "f1-score": 0.5824685703982012,
    "support": 724.0
  },
  "weighted avg": {
    "precision": 0.5883589769000775,
    "recall": 0.5883977900552486,
    "f1-score": 0.5853014599625282,
    "support": 724.0
  }
}
```