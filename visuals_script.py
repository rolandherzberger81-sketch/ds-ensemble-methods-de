import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np


def distribution(data, transformed=False):
    """
    Visualisierungscode zur Darstellung schiefer Verteilungen von Features
    """

    # Abbildung erstellen
    fig = pl.figure(figsize=(11, 5))

    # Darstellung der schiefen Features
    for i, feature in enumerate(["capital-gain", "capital-loss"]):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.hist(data[feature], bins=25, color="#00A0A0")
        ax.set_title("'%s' Feature Distribution" % (feature), fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels(["0", "500", "1000", "1500", ">2000"])

    # Plot-Ästhetik
    if transformed:
        fig.suptitle(
            "Log-transformed Distributions of Continuous Census Data Features",
            fontsize=16,
            y=1.03,
        )
    else:
        fig.suptitle(
            "Skewed Distributions of Continuous Census Data Features",
            fontsize=16,
            y=1.03,
        )

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualisierungscode zur Anzeige der Ergebnisse verschiedener Learner.

    inputs:
      - learners: Eine Liste von Algorithmen für überwachtes Lernen
      - stats: Eine Liste von Dictionaries mit den statistischen Ergebnissen aus 'train_predict()'
      - accuracy: Der Score für das naive Vorhersagemodell
      - f1: Der Score für das naive Vorhersagemodell
    """

    # Abbildung erstellen
    fig, ax = pl.subplots(2, 3, figsize=(12, 10))

    # Konstanten
    bar_width = 0.3
    colors = ["#A00000", "#00A0A0", "#00A000"]

    # Äußere Schleife zur Darstellung von vier Teilplots mit Daten
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(
            ["train_time", "acc_train", "f_train", "pred_time", "acc_test", "f_test"]
        ):
            for i in np.arange(3):
                # Kreativer Plot-Code
                ax[j // 3, j % 3].bar(
                    i + k * bar_width,
                    results[learner][i][metric],
                    width=bar_width,
                    color=colors[k],
                )
                ax[j // 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j // 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 3, j % 3].set_xlabel("Training Set Size")
                ax[j // 3, j % 3].set_xlim((-0.1, 3.0))

    # Eindeutige y-Labels hinzufügen
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Titel hinzufügen
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Horizontale Linien für die naiven Vorhersagemodelle hinzufügen
    ax[0, 1].axhline(
        y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    ax[1, 1].axhline(
        y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    ax[0, 2].axhline(
        y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    ax[1, 2].axhline(
        y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )

    # y-Grenzen für die Score-Teilplots festlegen
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Patches für die Legende erstellen
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    pl.legend(
        handles=patches,
        bbox_to_anchor=(-0.80, 2.53),
        loc="upper center",
        borderaxespad=0.0,
        ncol=3,
        fontsize="x-large",
    )

    # Ästhetik
    pl.suptitle(
        "Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10
    )
    pl.tight_layout()
    pl.show()


def feature_plot(importances, X_train, y_train):
    # Die fünf wichtigsten Features anzeigen
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Plot erstellen
    pl.figure(figsize=(9, 5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    pl.bar(
        np.arange(5),
        values,
        width=0.6,
        align="center",
        color="#00A000",
        label="Feature Weight",
    )
    pl.bar(
        np.arange(5) - 0.3,
        np.cumsum(values),
        width=0.2,
        align="center",
        color="#00A0A0",
        label="Cumulative Feature Weight",
    )
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize=12)
    pl.xlabel("Feature", fontsize=12)

    pl.legend(loc="upper center")
    pl.tight_layout()
    pl.show()
