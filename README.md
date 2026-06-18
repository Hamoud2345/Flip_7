# Flip 7 - Moteur de simulation et Stratégies optimales (MDP)

Ce dépôt contient un moteur de simulation pour le jeu de cartes Flip 7 (version calibrée à 79 cartes Numéro et 3 cartes Seconde Chance) ainsi que l'implémentation de stratégies d'IA basées sur la programmation dynamique et les processus de décision markoviens (MDP).

L'objectif est double :
1. Simuler fidèlement les règles du jeu (notamment la gestion fine de la défausse et du recyclage en cours de manche).
2. Calculer des stratégies optimales, d'abord à l'échelle d'une manche (maximisation de l'espérance de points), puis à l'échelle de la partie (maximisation de la probabilité de victoire globale).

## Structure du projet

* `flip7_game.py` : Le moteur de jeu. Il gère l'état des joueurs, le déroulement des manches, la pioche, et applique strictement la règle de non-recyclage des cartes de la table avant la fin de la manche (zone tampon intermédiaire).
* `Strategy.py` : Les algorithmes de décision. Contient la DP exacte pour l'espérance locale et la résolution de l'équation d'optimalité de Bellman (Value Iteration) pour la probabilité de gain globale.

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
Le projet nécessite Python 3.8+ et la bibliothèque NumPy.
```bash
pip install numpy
