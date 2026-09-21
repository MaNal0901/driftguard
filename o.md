
# DriftGuard — Plateforme MLOps de surveillance du drift

## 1. Présentation du projet

**DriftGuard** est une plateforme MLOps destinée à assurer la **surveillance continue des modèles de Machine Learning après leur déploiement**.

Lorsqu'un modèle est entraîné, il fonctionne généralement sur des données similaires à celles utilisées pendant son apprentissage. Cependant, les données réelles peuvent évoluer avec le temps. Cette évolution peut provoquer une diminution des performances du modèle.

DriftGuard a donc pour objectif de détecter automatiquement ces changements et d'aider à maintenir la fiabilité du modèle en production.

---

## 2. Problématique

Un modèle de Machine Learning n'est pas nécessairement performant indéfiniment.

Par exemple, un modèle de détection d'attaques réseau peut être entraîné sur des données historiques :

```text
Données historiques
       ↓
Entraînement
       ↓
Modèle ML
       ↓
Production
       ↓
Nouvelles données
```

Avec le temps, les caractéristiques du trafic réseau peuvent changer.

```text
Données Training ≠ Données Production
                 ↓
             Data Drift
                 ↓
       Performance du modèle ↓
```

Le problème est donc :

> **Comment surveiller automatiquement un modèle ML en production, détecter les changements dans les données et déclencher les actions nécessaires lorsque ses performances se dégradent ?**

---

# 3. Objectifs

Le projet DriftGuard aura plusieurs objectifs :

### Objectif principal

Développer une plateforme permettant de :

- surveiller les données en production ;
    
- détecter le **data drift** ;
    
- surveiller les performances du modèle ;
    
- détecter une éventuelle dégradation ;
    
- générer des alertes ;
    
- conserver l'historique des expériences et versions ;
    
- déclencher éventuellement un **réentraînement automatique**.
    

### Objectifs MLOps

Mettre en place une chaîne complète :

```text
Data
 ↓
Training
 ↓
Experiment Tracking
 ↓
Model Registry
 ↓
Deployment
 ↓
Monitoring
 ↓
Drift Detection
 ↓
Alert
 ↓
Retraining
 ↓
New Model
```

---

# 4. Architecture proposée

```text
                 ┌─────────────────┐
                 │     Dataset     │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │   Data Version  │
                 │      DVC        │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Model Training  │
                 │   Python/ML     │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │     MLflow      │
                 │ Tracking/Registry│
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  FastAPI Model  │
                 └────────┬────────┘
                          ↓
                ┌──────────────────┐
                │ Production Data  │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │    DriftGuard    │
                │                  │
                │ Data Drift       │
                │ Performance      │
                │ Monitoring       │
                └───────┬──────────┘
                        ↓
                 ┌──────────────┐
                 │    Alert     │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │ Retraining ? │
                 └──────┬───────┘
                        ↓
                    New Model
```

---

# 5. Technologies

Pour une première version, je te conseille :

|Besoin|Technologie|
|---|---|
|Machine Learning|Python + Scikit-learn|
|Data Versioning|**DVC**|
|Experiment Tracking|**MLflow**|
|API|**FastAPI**|
|Containerisation|**Docker**|
|Drift Detection|**Evidently**|
|Monitoring|**Prometheus + Grafana**|
|CI/CD|**GitHub Actions**|
|Code Versioning|**Git/GitHub**|

---

# 6. Cas d'utilisation proposé

Comme tu t'intéresses à la **cybersécurité**, on peut faire DriftGuard autour d'un modèle de **détection d'attaques réseau**.

### Dataset

**CICIDS2017**

Le modèle apprend à distinguer :

```text
Trafic réseau
      ↓
Normal / Attack
```

Par exemple :

```text
Normal
DDoS
PortScan
Brute Force
Bot
...
```

Puis le modèle est déployé avec FastAPI.

DriftGuard surveille ensuite les nouvelles données.

### Exemple

Pendant l'entraînement :

```text
Training:
Flow Duration
Packet Length
Destination Port
Flow Bytes/s
...
```

En production, la distribution peut changer :

```text
Training distribution
        ≠
Production distribution
        ↓
     DRIFT !
```

DriftGuard génère alors une alerte.

---

# 7. Fonctionnement du système

### Phase 1 — Training

```text
CICIDS2017
     ↓
Preprocessing
     ↓
Feature Engineering
     ↓
Training
     ↓
Model
```

### Phase 2 — Versioning

```text
Dataset → DVC
Code    → Git
Model   → MLflow
```

### Phase 3 — Deployment

```text
Model
  ↓
FastAPI
  ↓
Docker
  ↓
API
```

### Phase 4 — Monitoring

DriftGuard récupère les nouvelles données :

```text
Production data
       ↓
Drift Detection
       ↓
┌──────────────────┐
│ Drift ?          │
│ Performance ?    │
│ Latency ?        │
└────────┬─────────┘
         ↓
       Alert
```

### Phase 5 — Retraining

Si le drift est important ou si les performances diminuent :

```text
Alert
 ↓
Retraining
 ↓
Model v2
 ↓
MLflow
 ↓
Deployment
```

---

# 8. Première version du projet (MVP)

Je te conseille de **ne pas commencer directement avec toute l'architecture**.

Commence avec un MVP :

### Étape 1

Choisir le dataset.

➡️ **CICIDS2017**

### Étape 2

Créer un modèle simple.

Par exemple :

➡️ Random Forest ou XGBoost.

### Étape 3

Créer deux datasets :

```text
data/
├── reference/
│   └── train.csv
│
└── production/
    └── current.csv
```

### Étape 4

Créer une détection de drift avec **Evidently**.

### Étape 5

Ajouter MLflow.

### Étape 6

Conteneuriser avec Docker.

### Étape 7

Ajouter FastAPI.

### Étape 8

Ajouter Prometheus/Grafana.

### Étape 9

Ajouter le réentraînement automatique.

---

## 9. Résultat final attendu

À la fin, tu dois pouvoir montrer quelque chose comme :

```text
                 DRIFTGUARD
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
   Data Drift   Model Metrics   API Metrics
       │             │             │
       └─────────────┼─────────────┘
                     ↓
                  Dashboard
                     │
                     ↓
                  ⚠ Alert
                     │
                     ↓
               Retraining
                     │
                     ↓
                 Model v2
```

### Titre que tu peux utiliser

> **DriftGuard: An MLOps Platform for Continuous Monitoring and Automatic Retraining of Machine Learning Models**

Et pour commencer concrètement, je te conseille de faire **Étape 1 : préparer CICIDS2017 + définir exactement le modèle + créer la structure Git/DVC du projet**.