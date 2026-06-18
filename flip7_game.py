"""
Flip 7 (version réduite : 79 cartes numéro + 3 Seconde Chance) — moteur de jeu.

Constantes attendues depuis la cellule de préliminaires :
    NUMBERS      = list(range(13))
    FULL_NUM     = (1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)   # FULL_NUM[v] = copies du numéro v
    FULL_SC      = 3
    FLIP7_BONUS  = 15
    WIN_SCORE    = 200

Contrat avec les stratégies (laissé volontairement stable) : une stratégie est un objet
exposant `decide_hit(view) -> bool` ; elle lit `view.me`, `view.players`, `view.deck`,
`view.sc`, et sur son joueur `view.me.held`, `view.me.sc`, `view.me.score`. Elle peut aussi
exposer `choose_sc_recipient(view, eligible) -> Player` pour placer une Seconde Chance en trop.
"""

import random
from dataclasses import dataclass, field

# Types de carte renvoyés par Game._draw (évite les chaînes magiques disséminées).
NUMBER = "number"
SECOND_CHANCE = "second_chance"
NUMBERS  = list(range(13))                              # 0..12
FULL_NUM = tuple(1 if v == 0 else v for v in NUMBERS)   # un 0, un 1, deux 2, ..., douze 12
FULL_SC  = 3                                            # 3 Seconde Chance
FLIP7_BONUS, WIN_SCORE = 15, 200
assert sum(FULL_NUM) == 79

@dataclass
class Player:
    """État d'un joueur : son score cumulé et sa main de la manche en cours.

    Les noms held / sc / score / active / busted font partie du contrat lu par les
    stratégies via la View ; on les garde donc stables.
    """
    name: str
    strat: object                              # objet avec .decide_hit(view) -> bool
    score: int = 0                             # score cumulé sur toute la partie
    held: set = field(default_factory=set)     # numéros distincts accumulés cette manche
    sc: int = 0                                # 1 si le joueur détient une Seconde Chance
    active: bool = True                        # tire encore cette manche (ni arrêté ni busté)
    busted: bool = False                       # a pioché un doublon sans Seconde Chance
    # Cartes posées DEVANT le joueur pendant la manche (doublons bustés/absorbés, SC
    # consommées ou non données). Règle p.7 : elles restent là jusqu'à la fin de la
    # manche et ne sont donc JAMAIS remélangées en cours de manche.
    spent: list = field(default_factory=lambda: [0] * 13)
    discarded_sc: int = 0
    def reset(self):
        """Remet l'état de manche à zéro (le score cumulé est conservé)."""
        self.held = set()
        self.sc = 0
        self.active = True
        self.busted = False
        self.spent = [0] * 13
        self.discarded_sc = 0

    def points(self):
        """Points banqués cette manche : 0 si busté, sinon somme des numéros (+15 si Flip 7)."""
        if self.busted:
            return 0
        flip7_bonus = FLIP7_BONUS if len(self.held) == 7 else 0
        return sum(self.held) + flip7_bonus


@dataclass
class View:
    """Photo en lecture seule remise à une stratégie au moment de décider stop/hit.

    Expose exactement ce qu'un joueur peut légitimement voir : lui-même, tous les
    joueurs (pour leurs scores et tableaux visibles) et le deck restant exact.
    """
    me: Player
    players: list
    deck: list          # deck[v] = copies du numéro v encore dans le deck
    sc: int             # cartes Seconde Chance encore dans le deck

    def opp(self):
        """Les autres joueurs."""
        return [p for p in self.players if p is not self.me]


class Game:
    """Une partie de Flip 7 réduit : des manches jusqu'à ce qu'un score atteigne 200.

    Le deck (numéros + Seconde Chance) persiste d'une manche à l'autre et n'est
    remélangé depuis la défausse que lorsqu'il est épuisé.
    """

    def __init__(self, players, seed=None):
        self.players = players
        self.rng = random.Random(seed)
        # Deck vivant et défausse (numéros suivis par valeur, SC suivies par un compteur).
        self.deck = list(FULL_NUM)
        self.sc = FULL_SC
        self.discard = [0] * 13
        self.discard_sc = 0

    # ----- gestion du deck --------------------------------------------------
    def _cards_left(self):
        return sum(self.deck) + self.sc

    def _reshuffle_if_empty(self):
        """Deck épuisé : la défausse devient le nouveau deck."""
        if self._cards_left() == 0 and (sum(self.discard) + self.discard_sc) > 0:
            self.deck, self.sc = self.discard, self.discard_sc
            self.discard, self.discard_sc = [0] * 13, 0

    def _draw(self):
        """Pioche une carte uniformément. Renvoie (type, valeur) ou None.

        type vaut NUMBER (valeur = le numéro) ou SECOND_CHANCE (valeur = None).
        """
        self._reshuffle_if_empty()
        if self._cards_left() == 0:
            return None
        r = self.rng.randrange(self._cards_left())
        if r < self.sc:                        # les SC occupent les `sc` premières positions
            self.sc -= 1
            return (SECOND_CHANCE, None)
        r -= self.sc
        cumulative = 0
        for value in NUMBERS:
            cumulative += self.deck[value]
            if r < cumulative:
                self.deck[value] -= 1
                return (NUMBER, value)

    def _give_or_discard_sc(self, donor):
        """Seconde Chance piochée en trop : elle doit aller à un autre joueur encore
        en manche et sans SC ; la stratégie du donneur peut choisir le receveur."""
        eligible = [q for q in self.players
                    if q is not donor and q.active and not q.sc]
        if not eligible:                       # personne d'éligible -> reste devant le donneur
            donor.discarded_sc += 1            # (versée en défausse seulement en fin de manche)
            return
        chooser = getattr(donor.strat, "choose_sc_recipient", None)
        recipient = chooser(View(donor, self.players, self.deck, self.sc), eligible) if chooser else None
        if recipient not in eligible:          # défaut / sécurité : au plus petit score
            recipient = min(eligible, key=lambda q: q.score)
        recipient.sc = 1

    # ----- résolution d'une carte piochée -----------------------------------
    def _resolve_drawn_card(self, player, kind, value):
        """Applique la carte piochée à `player`. Renvoie True ssi c'était un Flip 7."""
        if kind == SECOND_CHANCE:
            if not player.sc:
                player.sc = 1
            else:
                self._give_or_discard_sc(player)
            return False

        # kind == NUMBER
        if value in player.held:               # doublon
            if player.sc:                      # la Seconde Chance l'absorbe
                player.sc = 0
                player.discarded_sc += 1       # SC consommée : reste devant le joueur
            else:                              # sinon : bust
                player.active = False
                player.busted = True
            player.spent[value] += 1           # doublon posé devant le joueur, pas en défausse
            return False

        player.held.add(value)                 # nouveau numéro distinct
        return len(player.held) == 7           # Flip 7 -> fin immédiate de la manche

    # ----- une manche -------------------------------------------------------
    def play_round(self, start):
        """Joue une manche en commençant au joueur d'indice `start`.

        Renvoie le joueur qui a fait Flip 7, ou None.
        """
        for p in self.players:
            p.reset()
        self._reshuffle_if_empty()

        n = len(self.players)
        order = [(start + i) % n for i in range(n)]
        flip7_player = None

        while True:
            self._reshuffle_if_empty()
            anyone_active = False
            for i in order:
                player = self.players[i]
                if not player.active:
                    continue
                anyone_active = True
                self._reshuffle_if_empty()    # Vérifie si le deck est vide avant de piocher
                view = View(player, self.players, self.deck, self.sc)
                if not player.strat.decide_hit(view):      # le joueur s'arrête (banque)
                    player.active = False
                    continue

                card = self._draw()
                if card is None:                           # plus rien à piocher
                    player.active = False
                    continue

                kind, value = card
                if self._resolve_drawn_card(player, kind, value):
                    flip7_player = player
                    break                                  # Flip 7 : on sort de la manche

            if flip7_player or not anyone_active:
                break

        self._bank_and_clear_round()
        return flip7_player

    def _bank_and_clear_round(self):
        """Ajoute le score de manche de chacun, puis envoie ses cartes à la défausse."""
        for p in self.players:
            # 1. ON BANQUE LES POINTS D'ABORD 
            p.score += p.points()

        for p in self.players:   
            # 2. On envoie la main en défausse globale
            for value in p.held:
                self.discard[value] += 1
                
            # 3. On envoie les doublons/cartes de la table en défausse globale
            for value in range(13):
                self.discard[value] += p.spent[value]
                
            # 4. On gère les Secondes Chances (restantes et consommées)
            if p.sc:
                self.discard_sc += 1
            self.discard_sc += p.discarded_sc
            
            # 5. On remet à zéro pour la manche suivante
            p.held = set()
            p.sc = 0
            p.spent = [0] * 13
            p.discarded_sc = 0

    # ----- partie entière ---------------------------------------------------
    def play(self, max_rounds=300):
        """Enchaîne les manches jusqu'à 200 points. Renvoie l'indice du vainqueur.

        Les égalités au sommet sont tranchées au hasard (simplification : la règle
        réelle joue des manches supplémentaires jusqu'à un meneur unique).
        """
        start = self.rng.randrange(len(self.players))
        for _ in range(max_rounds):
            self.play_round(start)
            start = (start + 1) % len(self.players)
            if any(p.score >= WIN_SCORE for p in self.players):
                break
        best = max(p.score for p in self.players)
        winners = [i for i, p in enumerate(self.players) if p.score == best]
        return self.rng.choice(winners) if len(winners) > 1 else winners[0]


#=================================
# Play Live
#=================================
class LiveGame(Game):
    """Identique à Game (même résultat, même probabilités) : ajoute juste le commentaire en direct."""

    def _reshuffle_if_empty(self):
        was_empty = self._cards_left() == 0
        super()._reshuffle_if_empty()
        if was_empty and self._cards_left() > 0:
            print("   (deck épuisé -> on remélange la défausse)")

    def _resolve_drawn_card(self, player, kind, value):
        already_had_sc = bool(player.sc)
        pre_held_size = len(player.held)
        flip7 = super()._resolve_drawn_card(player, kind, value)
        if kind == SECOND_CHANCE:
            msg = "pioche une 2e Seconde Chance (en trop)" if already_had_sc else "pioche une Seconde Chance"
            print(f"   -> {player.name} {msg}.")
        elif player.busted:
            print(f"   -> {player.name} pioche {value} (doublon) : BUST, manche perdue.")
        elif flip7:
            print(f"   -> {player.name} pioche {value} : FLIP 7 !! (+{player.points()} pts cette manche)")
        elif len(player.held) == pre_held_size:        # doublon absorbé par la SC
            print(f"   -> {player.name} pioche {value} (doublon), Seconde Chance utilisée.")
        else:
            print(f"   -> {player.name} pioche {value}, nouvelle carte.")
        return flip7

    def _bank_and_clear_round(self):
        snapshot = {p.name: (p.points(), sorted(p.held)) for p in self.players}
        super()._bank_and_clear_round()
        print("\n  Fin de manche :")
        for p in self.players:
            pts, held = snapshot[p.name]
            print(f"    {p.name:10s} +{pts:3d} pts (main {held or '(vide)'})  -> total {p.score}")

    def play_round(self, start):
        self._round_no = getattr(self, "_round_no", 0) + 1
        print(f"\n========== Manche {self._round_no} ==========")
        return super().play_round(start)


