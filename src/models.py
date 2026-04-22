from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def train_all_models(X, y):

    models = {}

    models['Logistic Regression'] = LogisticRegression(
        max_iter=300,
        class_weight='balanced'
    )

    models['SVM'] = SVC(
        kernel='linear',
        probability=True,
        class_weight='balanced'
    )

    models['Naive Bayes'] = MultinomialNB()

    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )

    for name, model in models.items():
        model.fit(X, y)

    return models