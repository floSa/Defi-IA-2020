# Défi IA — Prédiction des upvotes Reddit (TSE / UT3–UT1, 2020)

> Réécriture structurée du brief officiel (`Informations_Defi.txt`). Source de vérité pour
> la définition de la tâche, la métrique et le format de soumission.

## Problème
Peut-on prédire le nombre de votes positifs (`ups`) reçus par un commentaire Reddit à partir
de son contenu et de quelques méta-données ? **Problème de régression.**

Le sujet impose de **combiner analyse de réseaux (network mining) et analyse de texte
(text mining)**. La notation pédagogique récompense explicitement les deux volets
(30 pts text mining + 30 pts network mining), donc notre pipeline doit produire des features
riches dans *chacune* de ces deux familles.

## Données — `comments_students.csv`
Tous les commentaires du subreddit **AskReddit**, **mai 2015**. Un commentaire par ligne,
10 colonnes :

| Colonne | Description |
|---|---|
| `created_utc` | Date de création (epoch seconds UTC) |
| `ups` | **Cible.** Score du commentaire. `NaN` ⇒ ligne de test |
| `subreddit_id` | Id du subreddit (constant : `t5_2qh1i`) |
| `link_id` | Id du lien/submission (le « thread ») — préfixe `t3_` |
| `name` | Fullname du commentaire, ex. `t1_cqug90j` |
| `subreddit` | Nom du subreddit (constant : `AskReddit`) |
| `id` | Identifiant du commentaire, ex. `cqug90j` |
| `author` | Nom de compte de l'auteur (`[deleted]` fréquent) |
| `body` | Texte brut (markdown non nettoyé ; `<`, `>`, `&` échappés) |
| `parent_id` | Id de la chose à laquelle on répond (lien `t3_` ou commentaire `t1_`) |

**Split train/test** : les lignes avec `ups = NaN` constituent le jeu de test ; les autres le
jeu d'apprentissage. Ce découpage est imposé (pas de re-split de notre part).

## Métrique
**MAE** (Mean Absolute Error), **à minimiser**. Kaggle sépare le test en public (30 %) /
privé (70 %) ; le classement final est sur le privé.

> Note statistique clé : la MAE est minimisée par la **médiane** conditionnelle, pas la moyenne.
> Ici médiane(`ups`) = 1 et 52 % des commentaires valent exactement 1 → toute stratégie doit
> partir de ce baseline et se concentrer sur la queue (viral) et les downvotes.

## Format de soumission
```
id,predicted
id_1,prediction_1
...
```
`id` = colonne `id` du commentaire, `predicted` = `ups` prédit. En-tête `id,predicted`
**obligatoire**.

## Règles
- Ressources externes autorisées pour *apprendre* (Wikipedia, modèles pré-entraînés, etc.).
- **Interdit** : télécharger l'intégralité du jeu de données Reddit d'origine et l'utiliser
  (⇒ note de 0). Utiliser un modèle pré-entraîné public reste autorisé.
- Livrable attendu à l'époque : soumission Kaggle + archive de code commenté + `README`.

## Barème d'origine (contexte)
- 20 pts : score / classement Kaggle (reproductible).
- 60 pts : qualité de l'approche (30 text mining + 30 network mining), code commenté, README.
- 20 pts : organisation et clarté du code.
