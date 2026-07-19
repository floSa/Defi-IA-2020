# Préconisations pour la suite

> Écrit après la première soumission Kaggle réelle, qui a révélé un écart de **0,22 point**
> entre notre validation et le score privé. Ce document part de ce constat : tant qu'il n'est
> pas réglé, tout gain mesuré chez nous est invérifiable.

## 1. Le constat qui commande tout le reste

| Mesure | MAE |
|---|---|
| Notre holdout temporel (18-24 mai) | **7,8695** |
| Score **privé** Kaggle (70 % du test) | **8,0878** |
| Score **public** Kaggle (30 % du test) | **8,2353** |
| 1ᵉʳ du classement 2020 (Canards Niortais) | 7,8271 |
| 3ᵉ du classement 2020 | 7,9432 |

Deux faits à regarder en face :

- **Notre validation surestime de 0,218.** C'est **plus que la somme de tous les gains** obtenus
  pendant la session GPU (0,038 + 0,038 + 0,027…). Autrement dit : notre instrument de mesure est
  moins précis que ce qu'il prétend mesurer.
- **On est derrière les équipes de 2020**, avec un GPU qu'elles n'avaient pas et six ans de
  méthodes supplémentaires. Le retard est réel, pas un artefact de format.

L'écart de **0,147 entre score public et privé** est lui aussi instructif : il mesure la variance
entre deux sous-ensembles du *même* jeu de test. Tout gain inférieur à ~0,15 mesuré sur une seule
partition est donc **indistinguable du bruit d'échantillonnage**.

## 2. Priorité absolue : réparer la validation

Toutes nos décisions — choix de features, dimensionnalité, seuils des deux étages, poids du
blend — ont été prises sur **une unique fenêtre de 7 jours**. C'est beaucoup d'ajustements sur
un seul jeu, et le surapprentissage de validation est l'explication la plus simple de l'écart.

### 2.1 Validation multi-fenêtres (à faire en premier)
Évaluer sur **au moins trois fenêtres temporelles** (p. ex. 4-10, 11-17, 18-24 mai) et ne retenir
que les gains **présents sur les trois**. Un vrai signal tient sur toutes ; un artefact ne tient
que sur celle qu'on a optimisée.

Coût : ~3× le temps d'évaluation, aucun GPU. **C'est le meilleur rapport valeur/coût du projet.**

### 2.2 Rejouer l'historique des « gains »
Chaque amélioration de la session doit être repassée au crible multi-fenêtres :

| Gain revendiqué | Δ holdout | À vérifier |
|---|---|---|
| Embeddings figés | −0,188 | probablement réel (gros) |
| Probabilité du fine-tuning | −0,031 | à confirmer |
| Deux étages | −0,040 | à confirmer |
| Blend | −0,022 | **suspect** : les poids sont ajustés sur le holdout |
| Règle deux étages sur le blend | −0,007 | **très suspect** : sous le bruit inter-partitions |

Les deux derniers sont ceux qui ont le plus de chances d'être du bruit : ils sont petits **et**
leurs paramètres sont réglés sur le jeu qui les évalue.

### 2.3 Budget de décisions
Se fixer une règle explicite : **un seul jeu de validation ne supporte qu'un nombre limité de
décisions**. Au-delà, il faut une fenêtre fraîche. Pratiquement : geler une fenêtre « jamais
touchée » servant uniquement d'arbitrage final, et ne l'ouvrir qu'une fois.

## 3. Ce qui est solidement établi (à conserver)

Ces résultats sont robustes parce qu'ils reposent sur des écarts importants ou sur des
vérifications hors échantillon explicites :

- **Le network mining domine le text mining.** Les features structurelles (position dans le fil,
  heure, taille du thread) écrasent le texte. Le signal textuel plafonne à une AUC de 0,62 là où
  les features réseau atteignent 0,79 sur la même question.
- **La MAE est une régression de queue.** 52 % des `ups` valent 1 ; le cœur de la distribution ne
  pèse presque rien. Tout se joue sur la prédiction de la viralité.
- **`e5-small-v2` suffit.** `gte-modernbert-base` (2025, 4,5× plus gros) donne la **même** MAE
  (7,9996 vs 7,9968) pour 5× le coût d'encodage et 3,7× plus d'arbres. Ne pas y revenir.
- **64 dimensions d'embeddings, pas plus.** Courbe en U nette : 16 → 8,026 · 32 → 7,997 ·
  64 → 7,997 · 128 → 8,033. La variance retenue par la SVD n'est pas le bon critère de choix.
- **Le fine-tuning paie par sa sortie, pas par ses représentations.** Ses embeddings sont
  *moins* bons que les génériques ; seule sa prédiction apporte. Contre-intuitif et vérifié.
- **Les features issues d'un modèle doivent être générées hors échantillon.** Sans validation
  croisée, elles sont optimistes de 0,04 à 0,08 d'AUC sur les lignes d'entraînement, et le GBM
  leur fait trop confiance.

## 4. Pistes classées par rapport promesse/coût

### Prioritaires (sans GPU)
1. **Validation multi-fenêtres** (§2.1) — prérequis à tout le reste.
2. **Ré-arbitrer les modèles existants** sur cette base : il est possible qu'un modèle *plus
   simple* généralise mieux que notre champion. Six soumissions existent déjà, aucune n'a été
   comparée sur le vrai test.
3. **Simplifier volontairement** : tester le blend à poids égaux plutôt qu'optimisés, et le
   deux étages sans la règle finale. Si le score Kaggle ne bouge pas, ces étapes n'apportaient
   rien de réel.

### Moyennes
4. **Validation croisée à 4-5 plis** pour les features de fine-tuning (au lieu de 2). Chaque
   modèle verrait 75-80 % des données au lieu de 50 %. Coût : 2 à 2,5× le GPU d'aujourd'hui.
5. **Optimiser les bornes des tranches** d'upvotes — le découpage actuel
   (1 / 2-3 / 4-10 / 11-50 / 51+) a été choisi à vue, jamais testé contre des alternatives.
6. **Features de graphe** sur l'arbre de réponses. `node2vec` avait échoué ; un GNN reste ouvert,
   et c'est l'axe « network mining » que le challenge valorise explicitement (30 points sur 60).

### À ne pas poursuivre
- **ModernBERT ou tout encodeur plus gros** : équivalence démontrée, coût multiplié.
- **Plus de dimensions d'embeddings** : dégradation démontrée.
- **Fine-tuning en régression L1 directe** : s'effondre sur la médiane, et *abîme* les
  représentations (8,075 contre 7,997 sans fine-tuning).
- **Micro-optimisations du pipeline d'assemblage** : les trois derniers gains valaient 0,040 puis
  0,022 puis 0,007, tous sous le bruit inter-partitions de 0,147.

## 5. Trois pièges de mesure rencontrés — à ne pas refaire

Ils ont chacun produit un chiffre faux qui a survécu plusieurs heures, et deux d'entre eux ont
failli faire abandonner une bonne idée.

| Piège | Symptôme | Correctif en place |
|---|---|---|
| **Run non convergé** | `best_iter` collé au plafond d'arbres | Garde-fou dans `cli.py` : alerte si `best_iter + patience > plafond` |
| **Constante figée** | Référence ou nb d'arbres codés en dur, devenus faux | Lecture systématique depuis les rapports JSON |
| **Fuite in-sample** | Feature bien meilleure sur les lignes d'entraînement | Génération en validation croisée |

**Règle générale qui les résume** : un chiffre juste au moment où on l'écrit peut devenir faux en
silence quand le contexte change. Toute valeur doit voyager avec les données qui l'ont produite.

## 6. Ce que ce projet n'a pas encore exploité

Le challenge note **30 points sur 60 pour le network mining**. Or notre travail des dernières
sessions a porté presque exclusivement sur le texte — qui est, par nos propres mesures, la partie
la moins informative du problème. Il y a un déséquilibre à corriger, autant pour le score que
pour l'évaluation pédagogique :

- graphe de réponses exploité seulement par des features locales (profondeur, nb d'enfants) ;
- pas de features de **communauté** (détection de clusters d'auteurs, densité du voisinage) ;
- pas de propagation d'information dans l'arbre (les `ups` d'un frère aîné informent-ils ?) ;
- la dynamique temporelle du fil est agrégée, jamais modélisée comme une séquence.
