# Ensemble-Methoden

Ein praxisnaher Einblick in das Ensemble Learning mit Python. Du kombinierst mehrere Modelle per Voting, Mittelung und Stacking, trainierst und tunst einen XGBoost-Klassifikator, vergleichst Klassifikationsalgorithmen auf einem echten Zensus-Datensatz und baust zum Schluss den AdaBoost-Algorithmus von Grund auf.

## Lernziele

Am Ende dieses Repositorys kannst du:

- Ensembles per Voting und Mittelung bauen, die mehrere Basisklassifikatoren kombinieren.
- einen Stacking-Klassifikator implementieren, sowohl mit scikit-learn als auch aus seinen einzelnen Schritten.
- einen XGBoost-Klassifikator trainieren und seine Hyperparameter tunen.
- mehrere Klassifikationsalgorithmen auf einem echten Datensatz vergleichen und den stärksten auswählen.
- Vorverarbeitungsschritte (Log-Transformationen, One-Hot Encoding, Skalierung) vor der Modellierung anwenden.
- den AdaBoost-Algorithmus von Grund auf implementieren.

## Lernpfad

Bearbeite die Notebooks der Reihe nach.

| Datei / Ordner | Beschreibung |
|---|---|
| [**1 - Modelle kombinieren**](1_voting_ensemble_methods.ipynb) | Klassifikatoren mit Max Voting, Soft Voting, Mittelung, gewichteter Mittelung und Stacking kombinieren. |
| [**2 - XGBoost**](2_applying_xgboost.ipynb) | Daten vorbereiten, einen XGBoost-Klassifikator trainieren und Vorhersagen für den Diabetes-Datensatz erstellen. Wenn du macOS nutzt, lies zuerst den [Warnhinweis zu OpenMP](#6-notebooks-öffnen) im Setup. |
| [**3 - Vergleich von Klassifikatoren**](3_comparison_classification_algorithms_exercise.ipynb) | Aufgabe: die Zensusdaten vorverarbeiten, mehrere Klassifikatoren vergleichen und den besten tunen. |
| [**4 - AdaBoost von Grund auf (optional)**](4_optional_adaboost_python.ipynb) | Den AdaBoost-Algorithmus Schritt für Schritt in reinem Python aufbauen. |

### Weitere Ordner und Dateien

| Datei / Ordner | Beschreibung |
|---|---|
| [**Data**](data.zip) | Die Datensätze, gebündelt als ZIP-Datei (entpack sie beim Setup). |
| [**Solutions**](solutions/) | Musterlösungen. |
| [**Visualisierungshilfen**](visuals_script.py) | In den Notebooks verwendete Hilfsfunktionen für Plots. |
| [**pyproject.toml**](pyproject.toml) | Projektkonfiguration und Abhängigkeiten. |
| [**uv.lock**](uv.lock) | Lock-Datei für die Abhängigkeiten. |

## Setup

Verwende [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.5 oder neuer
und Python 3.13.17 oder ein neueres Patch-Release von 3.13, wie in
[`.python-version`](.python-version) festgelegt.

> [!NOTE]
> In allen Schritten ist Text in spitzen Klammern wie `<repo-name>` ein **Platzhalter**. Ersetz ihn samt den Klammern `< >` durch deinen eigenen Wert. Aus `cd <repo-name>` wird zum Beispiel `cd ds-ensemble-methods`.

### 1. Repository aus dem Template erstellen

Klick auf GitHub auf **Use this template**.

Beim Erstellen des Repositorys:

- Leg dich selbst als **Owner** fest
- Wähl einen Namen für das Repository
- Deaktiviere **Include all branches**
- Klick auf **Create repository**

> [!IMPORTANT]
> Wenn du zu zweit oder in einer Gruppe arbeitest, erledigt nur **eine Person** diesen Schritt.

---

### 2. Collaborators hinzufügen (nur bei Paar- oder Gruppenarbeit)

Wenn du mit Teammitgliedern arbeitest:

1. Öffne das Repository auf GitHub
2. Geh zu **Settings → Collaborators**
3. Füg deine Teammitglieder als Collaborators hinzu
4. Teil den Link zum Repository mit deinem Team

Die Teammitglieder sollten die Einladung annehmen, bevor sie weitermachen.

---

### 3. Repository klonen

Kopiere die SSH-URL über den Button **Code** auf GitHub und führ dann Folgendes aus:

```bash
git clone <copied-ssh-url>
```

Die kopierte SSH-URL hat die Form `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. In den Projektordner wechseln und Abhängigkeiten installieren

Das installiert alle Abhängigkeiten und erstellt eine virtuelle Umgebung in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Daten entpacken

Die Datensätze sind in `data.zip` gebündelt. Entpack sie in einen Ordner `data/`, bevor du die Notebooks ausführst. Dieser Befehl verwendet die Umgebung aus `uv sync`:

```bash
uv run python -c "import zipfile; zipfile.ZipFile('data.zip').extractall()"
```

---

### 6. Notebooks öffnen

> [!NOTE]
> Achte darauf, VS Code im Stammverzeichnis des Projekts zu öffnen, damit es die von `uv sync` erstellte Umgebung automatisch erkennt.

Starte VS Code im Stammverzeichnis des Projekts:

```bash
code .
```

Öffne dann ein Notebook und wähl die von `uv sync` erstellte Python-Umgebung als Kernel aus.

> [!CAUTION]
> Nur unter macOS: `uv sync` installiert XGBoost, aber nicht die OpenMP-Laufzeitbibliothek des Systems, auf die es angewiesen ist. Wenn [Notebook 2](2_applying_xgboost.ipynb) meldet, dass `libxgboost.dylib` nicht geladen werden konnte, installiere die Laufzeitbibliothek einmalig mit `brew install libomp`. Die Wheels für Linux und Windows enthalten OpenMP, dort ist also kein zusätzlicher Schritt nötig. Den offiziellen Hinweis findest du in der [Installationsdokumentation von XGBoost](https://xgboost.readthedocs.io/en/stable/install.html).

## Quellen & weiterführende Literatur

- [**Scikit-learn: Ensemble Methods**](https://scikit-learn.org/stable/modules/ensemble.html): Benutzerhandbuch zu Voting, Stacking, Boosting und Bagging.
- [**Scikit-learn: StackingClassifier**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html): API-Referenz für den in Notebook 1 verwendeten Stacking-Estimator.
- [**XGBoost Documentation**](https://xgboost.readthedocs.io/en/stable/): Offizielle Dokumentation der Gradient-Boosting-Bibliothek.
- [**The Random Forest Algorithm (MLU-Explain)**](https://mlu-explain.github.io/random-forest/): Eine visuelle, intuitive Erklärung dafür, warum die Kombination vieler Modelle funktioniert.


