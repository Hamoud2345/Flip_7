"""
Suite de tests pour le moteur Flip 7 et les stratégies.

Couvre les invariants critiques que le code suppose mais ne vérifiait pas :
- conservation des cartes (79 numéros + 3 Seconde Chance) à tout instant ;
- comptage des points (bust, somme, bonus Flip 7) ;
- règle du non-recyclage de la défausse en cours de manche ;
- absorption d'un doublon par la Seconde Chance ;
- correction de la DP exacte sur des cas calculables à la main ;
- reproductibilité par graine (seed).

Lancer avec :  pytest -q
"""
import random
import pytest

from flip7_game import (
    Game, Player, View, FULL_NUM, FULL_SC, FLIP7_BONUS,
    NUMBER, SECOND_CHANCE,
)
from Strategy import (
    Myopic, Optimal, Threshold, make_dp, myopic_hit,
)

TOTAL_NUM = sum(FULL_NUM)          # 79
TOTAL_SC = FULL_SC                 # 3


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def count_all_numbers(game):
    """Total des cartes numéro présentes partout (deck + défausse + joueurs).

    `held` est un ensemble de valeurs distinctes : chaque numéro tenu = une carte
    (on compte len, pas la somme). `spent` recense les doublons posés devant le joueur.
    """
    total = sum(game.deck) + sum(game.discard)
    for p in game.players:
        total += len(p.held)            # une carte par numéro distinct tenu
        total += sum(p.spent)           # doublons posés devant le joueur
    return total


def count_all_sc(game):
    """Total des Seconde Chance présentes partout."""
    total = game.sc + game.discard_sc
    for p in game.players:
        total += p.sc + p.discarded_sc
    return total


# --------------------------------------------------------------------------
# 1. Conservation des cartes — l'invariant le plus important
# --------------------------------------------------------------------------
def test_deck_initial_counts():
    g = Game([Player("A", Myopic()), Player("B", Myopic())])
    assert sum(g.deck) == TOTAL_NUM
    assert g.sc == TOTAL_SC


def test_number_conservation_through_full_game():
    """Le nombre total de cartes numéro doit rester constant pendant toute la partie."""
    g = Game([Player("A", Optimal()), Player("B", Myopic()),
              Player("C", Threshold(25))], seed=12345)
    g.play(max_rounds=50)
    assert count_all_numbers(g) == TOTAL_NUM


def test_sc_conservation_through_full_game():
    """Idem pour les Seconde Chance."""
    g = Game([Player("A", Optimal()), Player("B", Myopic()),
              Player("C", Threshold(25))], seed=999)
    g.play(max_rounds=50)
    assert count_all_sc(g) == TOTAL_SC


@pytest.mark.parametrize("seed", range(8))
def test_conservation_invariant_after_each_round(seed):
    """Après chaque manche banquée, rien ne doit avoir été créé ni perdu."""
    g = Game([Player("A", Optimal()), Player("B", Myopic())], seed=seed)
    start = 0
    for _ in range(15):
        g.play_round(start)
        start = (start + 1) % 2
        assert count_all_numbers(g) == TOTAL_NUM
        assert count_all_sc(g) == TOTAL_SC


# --------------------------------------------------------------------------
# 2. Comptage des points
# --------------------------------------------------------------------------
def test_points_busted_is_zero():
    p = Player("X", Myopic())
    p.held = {3, 5, 9}
    p.busted = True
    assert p.points() == 0


def test_points_simple_sum():
    p = Player("X", Myopic())
    p.held = {2, 4, 7}
    assert p.points() == 13          # pas de Flip 7 (moins de 7 cartes)


def test_points_flip7_bonus():
    p = Player("X", Myopic())
    p.held = {1, 2, 3, 4, 5, 6, 7}   # 7 numéros distincts
    assert p.points() == 28 + FLIP7_BONUS   # somme 28 + 15


def test_points_six_cards_no_bonus():
    p = Player("X", Myopic())
    p.held = {1, 2, 3, 4, 5, 6}
    assert p.points() == 21          # pas encore le bonus


# --------------------------------------------------------------------------
# 3. Résolution d'une carte : Seconde Chance & bust
# --------------------------------------------------------------------------
def make_solo_game():
    return Game([Player("Solo", Myopic())])


def test_second_chance_absorbs_duplicate():
    g = make_solo_game()
    p = g.players[0]
    p.held = {5}
    p.sc = 1
    flip7 = g._resolve_drawn_card(p, NUMBER, 5)   # doublon
    assert flip7 is False
    assert p.busted is False                      # absorbé, pas de bust
    assert p.sc == 0                              # SC consommée
    assert p.spent[5] == 1                        # doublon posé devant le joueur
    assert p.discarded_sc == 1


def test_duplicate_without_sc_busts():
    g = make_solo_game()
    p = g.players[0]
    p.held = {5}
    p.sc = 0
    g._resolve_drawn_card(p, NUMBER, 5)
    assert p.busted is True
    assert p.active is False
    assert p.spent[5] == 1


def test_seventh_distinct_card_is_flip7():
    g = make_solo_game()
    p = g.players[0]
    p.held = {1, 2, 3, 4, 5, 6}
    flip7 = g._resolve_drawn_card(p, NUMBER, 7)
    assert flip7 is True
    assert len(p.held) == 7


def test_first_second_chance_is_kept():
    g = make_solo_game()
    p = g.players[0]
    p.sc = 0
    g._resolve_drawn_card(p, SECOND_CHANCE, None)
    assert p.sc == 1


# --------------------------------------------------------------------------
# 4. Règle du non-recyclage en cours de manche
# --------------------------------------------------------------------------
def test_no_reshuffle_while_cards_remain():
    """_reshuffle_if_empty ne doit rien faire tant que le deck n'est pas vide."""
    g = make_solo_game()
    deck_before = list(g.deck)
    sc_before = g.sc
    g.discard[3] = 10            # défausse non vide
    g.discard_sc = 1
    g._reshuffle_if_empty()
    assert g.deck == deck_before
    assert g.sc == sc_before     # le deck non vide n'a pas été remplacé


def test_reshuffle_when_empty():
    g = make_solo_game()
    g.deck = [0] * 13
    g.sc = 0
    g.discard = [0] * 13
    g.discard[4] = 2
    g.discard_sc = 1
    g._reshuffle_if_empty()
    assert g.deck[4] == 2        # la défausse est devenue le deck
    assert g.sc == 1
    assert sum(g.discard) == 0
    assert g.discard_sc == 0


# --------------------------------------------------------------------------
# 5. Correction de la DP exacte sur des cas calculables à la main
# --------------------------------------------------------------------------
def test_dp_empty_deck_returns_current_value():
    """Deck vide : la valeur d'un état est exactement la somme déjà en main."""
    deck = [0] * 13
    dp = make_dp(deck, 0)
    assert dp((3, 7), 0) == pytest.approx(10.0)


def test_dp_single_safe_card_is_taken():
    """Une seule carte restante, non-doublon : tirer ajoute sa valeur (sans risque)."""
    deck = [0] * 13
    deck[6] = 1                    # un seul 6 dans le deck
    dp = make_dp(deck, 0)
    # main {2} : tirer le 6 donne 8 et le deck devient vide -> valeur 8 > stand 2
    assert dp((2,), 0) == pytest.approx(8.0)


def test_dp_only_duplicates_left_no_sc_means_stand():
    """S'il ne reste que des doublons et aucune SC, tirer = bust certain -> on garde."""
    deck = [0] * 13
    deck[4] = 3                    # il ne reste que des 4
    dp = make_dp(deck, 0)
    # main {4} : tout tirage est un doublon -> bust (0). On doit donc rester à 4.
    assert dp((4,), 0) == pytest.approx(4.0)


def test_dp_value_is_at_least_stand_value():
    """La DP ne peut jamais valoir moins que de s'arrêter tout de suite."""
    g = Game([Player("A", Optimal())])
    dp = make_dp(tuple(g.deck), g.sc)
    held = (2, 5, 9)
    assert dp(held, 0) >= sum(held) - 1e-9


# --------------------------------------------------------------------------
# 6. Règle myope cohérente
# --------------------------------------------------------------------------
def test_myopic_hits_on_empty_hand():
    g = Game([Player("A", Myopic())])
    assert myopic_hit(set(), 0, g.deck, g.sc) is True   # aucun risque, on tire


def test_myopic_hits_when_holding_sc():
    deck = list(FULL_NUM)
    # même avec un doublon possible, une SC en réserve => on tire
    assert myopic_hit({3}, 1, deck, 0) is True


# --------------------------------------------------------------------------
# 7. Reproductibilité par graine
# --------------------------------------------------------------------------
def test_same_seed_same_winner():
    def run():
        g = Game([Player("A", Optimal()), Player("B", Myopic()),
                  Player("C", Threshold(20))], seed=2024)
        return g.play()
    assert run() == run()


def test_different_seeds_independent():
    """Au moins deux graines différentes ne donnent pas toutes le même déroulé."""
    finals = []
    for s in range(5):
        g = Game([Player("A", Optimal()), Player("B", Myopic())], seed=s)
        g.play()
        finals.append(tuple(p.score for p in g.players))
    assert len(set(finals)) > 1


# --------------------------------------------------------------------------
# 8. Intégration : une partie se termine et désigne un vainqueur valide
# --------------------------------------------------------------------------
def test_play_returns_valid_winner_index():
    players = [Player("A", Optimal()), Player("B", Myopic()), Player("C", Threshold(25))]
    g = Game(players, seed=7)
    w = g.play()
    assert 0 <= w < len(players)


def test_view_opp_excludes_self():
    players = [Player("A", Myopic()), Player("B", Myopic())]
    g = Game(players)
    v = View(players[0], players, g.deck, g.sc)
    assert players[0] not in v.opp()
    assert players[1] in v.opp()
