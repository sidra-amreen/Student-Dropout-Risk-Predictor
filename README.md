# Student Dropout Risk Predictor

A machine learning project that estimates each student's probability of dropping out and
explains which factors drive the risk, so advisors can intervene early.

## Setup
```bash
pip install -r requirements.txt
python train.py      # generates data (if missing), trains 3 models, saves model + results.png
python predict.py    # scores 3 example students
python predict.py student.json   # score your own student
```

## Pipeline
1. `generate_data.py` - synthetic data with 13 features: academics (GPA, failed courses, credits passed),
   engagement (attendance, assignments, LMS logins), finances (scholarship, tuition, income) and context
   (age, part-time work, first-generation, commute).
2. `train.py` - compares Logistic Regression, Random Forest and Gradient Boosting with 5-fold
   cross-validated ROC-AUC, picks the best, and evaluates with a 0.4 threshold (favoring recall, because
   missing an at-risk student is costlier than a false alarm). Produces ROC curves, a confusion matrix
   and permutation feature importance in `results.png`.
3. `predict.py` - outputs probability, a LOW / MEDIUM / HIGH tier, and the top factors raising or
   lowering that student's risk (from the interpretable logistic regression).

## Using real data
Replace `data/students.csv` with real records containing the columns in `features.py` plus `dropout` (0/1).
A good public option is the UCI "Predict Students' Dropout and Academic Success" dataset (map its columns
to `features.py` or edit `FEATURES`).

## Responsible use
Predictions should support, not replace, human judgment. Use them to offer tutoring, financial
counseling or mentoring, never to deny opportunities. Audit for bias across groups (e.g. first-generation,
income level) before deploying, and be open with students about how the scores are used.
