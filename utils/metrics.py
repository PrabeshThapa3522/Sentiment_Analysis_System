import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
        y_true,
        y_pred,
        class_names,
        model_name="Model"
):

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    plt.figure(figsize=(7,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(
        model_name + " Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        f"results/{model_name.lower().replace(' ','_')}_confusion_matrix.png",
        dpi=300
    )

    plt.show()

    return {

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1

    }


if __name__ == "__main__":

    y_true = [
        0,1,2,
        0,1,2
    ]

    y_pred = [
        0,1,2,
        1,1,2
    ]

    class_names = [
        "negative",
        "neutral",
        "positive"
    ]

    evaluate_model(
        y_true,
        y_pred,
        class_names,
        "Sample Model"
    )