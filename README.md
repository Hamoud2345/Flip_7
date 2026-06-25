# Flip 7 - Moteur de simulation et Stratégies optimales (MDP)

Ce dépôt contient un moteur de simulation pour le jeu de cartes Flip 7 (version calibrée à 79 cartes Numéro et 3 cartes Seconde Chance) ainsi que l'implémentation de stratégies d'IA basées sur la programmation dynamique et les processus de décision markoviens (MDP).

L'objectif est double :
1. Simuler fidèlement les règles du jeu (notamment la gestion fine de la défausse et du recyclage en cours de manche).
2. Calculer des stratégies optimales, d'abord à l'échelle d'une manche (maximisation de l'espérance de points), puis à l'échelle de la partie (maximisation de la probabilité de victoire globale).

## Structure du projet

* `flip7_game.py` : Le moteur de jeu. Il gère l'état des joueurs, le déroulement des manches, la pioche, et applique strictement la règle de non-recyclage des cartes de la table avant la fin de la manche (zone tampon intermédiaire).
* `Strategy.py` : Les algorithmes de décision. Contient la DP exacte pour l'espérance locale et la résolution de l'équation d'optimalité de Bellman (Value Iteration) pour la probabilité de gain globale.
* `test_flip7.py` : La suite de tests unitaires (via Pytest) validant les invariants mathématiques et les règles du jeu.
* `requirements.txt` : La liste des dépendances du projet (NumPy et Pytest).

## Approche théorique et Modélisation

### 1. Optimisation locale (Échelle de la manche)
Pendant son tour, l'IA utilise une programmation dynamique (DP) exacte sans remise. À chaque étape, elle évalue si la distribution résiduelle du deck justifie de piocher (Hit) ou de s'arrêter (Stand), en prenant en compte le fait qu'une carte Seconde Chance possède un rôle d'amortisseur de doublon (Bust).

### 2. Optimisation globale (Échelle de la partie)
Pour maximiser la probabilité de gagner la partie (atteindre 200 points avant les autres), le problème est modélisé à un niveau macro :
* Une simulation de Monte Carlo préliminaire estime la fonction de masse (PMF) des incréments de score par manche (`fX`).
* Un algorithme d'itération sur la valeur (Value Iteration) résout le point fixe de Bellman sur un tenseur 3D représentant les scores des 3 joueurs.
* Les calculs de transition utilisent des opérations de corrélation croisée en avant (forward cross-correlation) vectorisées avec NumPy pour accélérer la convergence (critère d'arrêt sous la norme L-infini).


## Installation et Utilisation

### Prérequis
Installez l'ensemble des dépendances nécessaires via le fichier `requirements.txt` :
```bash
pip install -r requirements.txt
```

### Lancer la suite de tests
Les tests valident les invariants du moteur (conservation des cartes, comptage des points, règle de non-recyclage) et la correction de la DP exacte :
```bash
pytest -q
```

### Jouer une partie (stratégies locales)
Les stratégies `Optimal` (espérance), `Myopic` et `Threshold` s'utilisent directement, sans pré-calcul :
```python
from flip7_game import Game, Player
from Strategy import Optimal, Myopic, Threshold

players = [Player("A", Optimal()), Player("B", Myopic()), Player("C", Threshold(25))]
game = Game(players, seed=42)
winner = game.play()          # renvoie l'indice du joueur gagnant
print(winner, [p.score for p in game.players])
```

### Stratégie en probabilité de victoire (`WinProb`)
`WinProb` exige le tenseur de valeur global `W` et la distribution des incréments `fX`, tous deux
construits dans le notebook `Exec.ipynb` (estimation Monte-Carlo de `fX` puis Value Iteration sur
la grille des scores). Le calcul de `W` est lourd (tenseur `200³`, corrélations sur tout le support
de `fX`) : comptez plusieurs minutes. Une fois `W` et `fX` disponibles :
```python
from Strategy import WinProb
player = Player("WinProb", WinProb(W, fX))
```
Le notebook propose aussi une partie commentée en direct (`LiveGame`) et un mode interactif
humain contre IA (`Human`).

## Documentation théorique

La note `flip7_note.pdf` formalise le problème et les résultats utilisés ici : modélisation à deux
échelles de temps, équation d'optimalité de Bellman, théorème de réduction de la maximisation de
P(victoire) à un arrêt optimal modifié, et existence / unicité / convergence de l'itération sur la
valeur (transience, rayon spectral `< 1`, série de Neumann).
