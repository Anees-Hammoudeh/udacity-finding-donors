# Finding Donors for CharityML

Project from the **Udacity Intro to Machine Learning Nanodegree**. Uses supervised learning to predict whether a person earns more than $50,000 a year, based on 1994 U.S. Census data. The fictional charity *CharityML* would use this prediction to target people who are likely to donate.

## Approach
1. **Explore and preprocess:** log-transformed skewed features (`capital-gain`, `capital-loss`), scaled numeric features with MinMaxScaler, and one-hot encoded categorical features.
2. **Benchmark:** a naive predictor that always predicts ">50K" scores accuracy 0.2478 and F-score 0.2917.
3. **Compare models:** trained and evaluated three supervised models (including Random Forest and Logistic Regression) on 1%, 10% and 100% of the training data, using accuracy, F-score (β = 0.5) and training time.
4. **Tune:** optimized Logistic Regression with GridSearchCV over the regularization parameter.
5. **Feature importance:** found the most predictive features and retrained on only the top five.

## Results
| Metric | Naive predictor | Unoptimized model | Optimized model | Top-5 features only |
|:--|:--:|:--:|:--:|:--:|
| Accuracy | 0.2478 | 0.8417 | **0.8421** | 0.8294 |
| F-score (β = 0.5) | 0.2917 | 0.6826 | **0.6843** | 0.6548 |

The **most important features** were age, hours-per-week, capital-gain, education-num and marital status (married-civ-spouse).

## Files
| File | Description |
|---|---|
| `finding_donors.ipynb` | Main notebook |
| `report.html` | Rendered report |
| `visuals.py` | Plotting helpers provided by Udacity |
| `census.csv` | Census dataset |

## Run it
```bash
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook finding_donors.ipynb
```
