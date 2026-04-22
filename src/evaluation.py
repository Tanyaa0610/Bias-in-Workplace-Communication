import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)


# -------------------------
# BASELINE EVALUATION
# -------------------------
def evaluate_all(models, X_test, y_test, sem_test):

    results = {}
    roc_data = {}

    for name, model in models.items():

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = [acc, prec, rec, f1]

        # -------------------------
        # CONFUSION MATRIX (CLEAN)
        # -------------------------
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Bias','Bias'],
                    yticklabels=['Non-Bias','Bias'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(f"cm_{name}.png", bbox_inches='tight')
        plt.close()

        # -------------------------
        # ROC DATA
        # -------------------------
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        roc_data[name] = (fpr, tpr, roc_auc)

    # -------------------------
    # ROC CURVE (ALL)
    # -------------------------
    plt.figure(figsize=(6,5))

    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Baseline Models")
    plt.legend()
    plt.savefig("roc_baselines.png", bbox_inches='tight')
    plt.close()

    # -------------------------
    # PRECISION-RECALL CURVE
    # -------------------------
    plt.figure(figsize=(6,5))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:,1]

        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        plt.plot(recall, precision, label=name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.savefig("pr_curve.png", bbox_inches='tight')
    plt.close()

    return results


# -------------------------
# HYBRID MODEL EVALUATION
# -------------------------
def evaluate_hybrid(best_model, X_test, y_test, sem_test):

    y_prob = best_model.predict_proba(X_test)[:,1]

    sem_norm = (sem_test.values + 1) / 2
    final_score = 0.7*y_prob + 0.3*sem_norm

    y_pred = (final_score > 0.4).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # -------------------------
    # CONFUSION MATRIX HYBRID
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Non-Bias','Bias'],
                yticklabels=['Non-Bias','Bias'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Hybrid")
    plt.savefig("cm_hybrid.png", bbox_inches='tight')
    plt.close()

    # -------------------------
    # ROC HYBRID
    # -------------------------
    fpr, tpr, _ = roc_curve(y_test, final_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"Hybrid (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hybrid Model")
    plt.legend()
    plt.savefig("roc_hybrid.png", bbox_inches='tight')
    plt.close()

    return {"Hybrid": [acc, prec, rec, f1]}


# -------------------------
# METRICS + TABLE (CLEAN)
# -------------------------
def plot_metrics(results):

    df = pd.DataFrame(results, index=["Accuracy","Precision","Recall","F1"]).T

    # -------------------------
    # BAR GRAPH
    # -------------------------
    df.plot(kind='bar', figsize=(8,5))
    plt.title("Model Comparison")
    plt.xticks(rotation=30)
    plt.savefig("metrics_comparison.png", bbox_inches='tight')
    plt.close()

    # -------------------------
    # CLEAN TABLE IMAGE
    # -------------------------
    fig, ax = plt.subplots(figsize=(10,3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.round(2).values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.savefig("results_table_clean.png", bbox_inches='tight')
    plt.close()