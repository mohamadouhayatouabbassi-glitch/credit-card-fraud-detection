# Détection de Fraude par Carte Bancaire

## 📋 Résumé du Projet
Modèle de machine learning pour détecter les transactions frauduleuses sur un dataset fortement déséquilibré (0.17% de fraudes). Le projet compare deux approches et déploie un modèle **XGBoost** optimisé pour la production.

---

## 📊 Dataset
- **Source** : [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Taille** : 284 808 transactions
- **Features** : 30 colonnes (V1–V28 : composantes PCA, `Time`, `Amount`)
- **Cible** : `Class` (0 = normal, 1 = fraude)
- **Déséquilibre** : 99.83% de transactions normales, 0.17% de fraudes

---

## 🔍 Exploration & Préparation (Étape 1-2)
### Étape 1 : Analyse Exploratoire
- Distribution des classes (déséquilibre confirmé)
- Statistiques descriptives par classe
- Corrélations et détection d'anomalies
- **Résultat** : Dataset sain, peu de valeurs manquantes, déséquilibre majeur identifié

### Étape 2 : Préparation des Données
- Suppression des doublons exacts (3 lignes)
- Split stratifié : 60% train / 20% validation / 20% test
- **Résultat** : 170 235 (train) / 56 745 (val) / 56 746 (test) avec ratio fraude stable (~0.17%)

---

## 🎯 Modélisation (Étape 3-4)

### Étape 3 : Baseline — Régression Logistique
**Architecture :**
- Pipeline : StandardScaler (Amount) + LogisticRegression
- `class_weight="balanced"` pour gérer le déséquilibre

**Résultats (test, seuil calibré pour précision ≥ 0.90) :**
- PR-AUC : **0.678**
- Précision : 0.859
- Recall : **0.579** ⚠️ (40 fraudes ratées)
- Faux positifs : 9

**Conclusion** : modèle acceptable mais rappel insuffisant.

---

### Étape 4 : Modèle Principal — XGBoost
**Architecture :**
- XGBClassifier : 400 arbres, max_depth=4, learning_rate=0.05
- `scale_pos_weight=598.4` pour le déséquilibre
- `eval_metric="aucpr"` (optimisation sur PR-AUC)

**Résultats (test, seuil=0.7124 calibré pour précision ≥ 0.90) :**
- PR-AUC : **0.825** (+22% vs baseline)
- Précision : 0.905 ✅
- Recall : **0.800** (+38% vs baseline)
- Faux positifs : 8

**Conclusion** : **modèle sélectionné pour production** — meilleure détection de fraudes avec fausses alertes similaires.

---

## 📈 Comparaison Finale

| Métrique | Régression Logistique | XGBoost | Gain |
|----------|----------------------|---------|------|
| PR-AUC (test) | 0.678 | **0.825** | +21.7% |
| Recall (test) | 0.579 | **0.800** | +38.2% |
| Precision (test) | 0.859 | 0.905 | +5.4% |
| Fraudes détectées | 55/95 | **76/95** | +21 fraudes |
| Fraudes ratées | 40 | **19** | -52% erreurs |

**Avantage XGBoost :** À coût égal de fausses alertes, détecte 40% plus de fraudes.

---

## 📁 Structure du Projet
```
.
├── data/
│   └── creditcard.csv                 # Dataset brut (284k transactions)
├── notebooks/
│   ├── 01_exploration.ipynb           # Analyse exploratoire
│   ├── 02_preparation.ipynb           # Préparation données
│   ├── 03_baseline.ipynb              # Régression logistique
│   └── 04_xgboost.ipynb               # XGBoost + évaluation
├── models/
│   ├── xgboost_fraude.pkl             # Modèle sauvegardé (joblib)
│   └── metadata_xgboost.json          # Seuil + métadonnées
├── src/
│   └── scoring.py                     # Fonction de scoring (future API)
├── README.md                          # Ce fichier
├── .gitignore
└── LICENSE
```

---

## 🚀 Utilisation du Modèle

### Installation
```bash
pip install xgboost scikit-learn pandas numpy joblib
```

### Chargement & Prédiction
```python
from src.scoring import ScoringModel

model = ScoringModel()

# Score une seule transaction
resultat = model.score_transaction({
    'Time': 0, 'Amount': 149.62,
    'V1': -1.3598, 'V2': -0.0747,
    # ... (autres features V3–V28)
})

print(resultat)
# {'prediction': 0, 'probabilite': 0.024, 'seuil': 0.7124, 'est_fraude': False}
```

### Batch Scoring
```python
predictions = model.predire(X_test)  # Retourne 0/1
probas = model.predire_proba(X_test)  # Retourne probabilités
```

---

## 📝 Recommandations pour Production

1. **Seuil de décision** : 0.7124 (calibré pour précision ≥ 0.90)
   - Ajustable selon coûts métier (faux positif vs faux négatif)

2. **Monitoring** : tracker PR-AUC, précision, recall régulièrement
   - Réentraîner si drift > 5% sur métrique clé

3. **Features** : stabiliser le calcul des composantes PCA (V1–V28)
   - Documenter la transformation `Amount` (StandardScaler)

4. **API** : voir `src/scoring.py` pour wrapper FastAPI

---

## 🔧 Technologies
- **Python 3.10+**
- **scikit-learn** : preprocessing, modèles, évaluation
- **XGBoost** : gradient boosting
- **Pandas / NumPy** : manipulation données
- **Joblib** : sérialisation modèle

---

## 📚 Références
- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PR-AUC vs ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Handling Imbalanced Classification](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)

---

## 👤 Auteur
**mohamadouhayatouabbassi-glitch**  
Juin 2026

---

## 📄 Licence
MIT - Voir fichier `LICENSE` pour plus de détails
