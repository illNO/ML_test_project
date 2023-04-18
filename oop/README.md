`Classifier.py` contains the implementation of multiple inheritance.<br>

The `DigitClassificationInterface` interface has 3 child classes:
1. `CNNDigitClassifier` input shape [28, 28, 1]
2. `RandomForestDigitClassifier` input shape [784,]
3. `RandomDigitClassifier` input shape [10, 10]

Each class implements different model


`DigitClassifier` takes one of three input parameters: `cnn`, `rf`, `rand` - specifying type of model, that should be used

