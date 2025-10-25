| Rank | Model            | Best CV Macro-F1 | Best Params                                                                                   |
| ---: | ---------------- | ---------------: | --------------------------------------------------------------------------------------------- |
|    1 | **MLP**          |     **0.308294** | `{'mlpclassifier__alpha': 0.01, 'mlpclassifier__hidden_layer_sizes': (…)}`                    |
|    2 | **BernoulliNB**  |     **0.299671** | `{'bernoullinb__alpha': 0.1}`                                                                 |
|    3 | **DecisionTree** |     **0.279022** | `{'decisiontreeclassifier__max_depth': None, 'decisiontreeclassifier__min_samples_split': …}` |
|    4 | **GaussianNB**   |     **0.252555** | `{'gaussiannb__var_smoothing': 1e-07}`                                                        |
|    5 | **LogReg_L2**    |     **0.225567** | `{'logisticregression__C': 10.0}`                                                             |
