## Roadmap — vers un système de prévision professionnel

Ce projet utilise volontairement des features purement calendaires pour
garder le monitoring de drift simple et interprétable. Pour un vrai système
de production, les leviers suivants amélioreraient significativement la
précision (R² actuel : ~0.70 sur validation) :

| Priorité | Amélioration | Gain estimé |
|---|---|---|
| 1 | Données météo (température, historique + prévisions) | Le plus important — cause directe de la consommation |
| 2 | Lag features (J-1, J-7, moyennes glissantes) | Gros gain, mais complexifie l'interprétation du drift monitoring |
| 3 | Jours fériés + veilles de jours fériés | Gain modéré |
| 4 | Modèles spécialisés séries temporelles (Prophet, SARIMA, LSTM) | Gain modéré à important |
| 5 | Ré-entraînement périodique sur fenêtre glissante | Indispensable en vraie prod, pas de gain immédiat de précision |

Non implémenté ici pour garder le focus sur la démonstration du pipeline
MLOps (versioning, CI/CD, tracking, monitoring de drift) plutôt que sur
l'optimisation de la performance prédictive.