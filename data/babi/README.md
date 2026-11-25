# bAbI (en-10k), tâches 1, 2, 3 et 6

Données de Weston, Bordes, Chopra, Rush, Merriënboer, Joulin et Mikolov (2015), *Towards AI-complete question answering: a set of prerequisite toy tasks* (https://arxiv.org/abs/1502.05698), version 1.2, 10 000 questions d'entraînement par tâche, 1 000 de test. Licence : Creative Commons Attribution 3.0.

Copie fichier à fichier depuis `data/tasks_1-20_v1-2/en-10k/` du dépôt public github.com/ethancaballero/Improved-Dynamic-Memory-Networks-DMN-plus (le serveur d'origine de Facebook n'est pas joignable depuis le bac à sable). md5 de `qa1_single-supporting-fact_train.txt` : 1835e9ea97b02d6546071b41987bcd8b.

| fichier | tâche |
|---|---|
| qa1_single-supporting-fact | « Où est Mary ? » : 1 fait utile |
| qa2_two-supporting-facts | « Où est le ballon ? » : qui le tient, puis où est cette personne (2 faits) |
| qa3_three-supporting-facts | « Où était le ballon avant la cuisine ? » : 3 faits |
| qa6_yes-no-questions | « Mary est-elle dans le jardin ? » : oui / non |

Utilisées par l'amendement 22 (`scripts/cpu_babi.py`).
