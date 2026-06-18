import random, math, statistics
import numpy as np
# Constants
NUMBERS  = list(range(13))                              # 0..12
FULL_NUM = tuple(1 if v == 0 else v for v in NUMBERS)   # un 0, un 1, deux 2, ..., douze 12
FULL_SC  = 3                                            # 3 Seconde Chance
FLIP7_BONUS, WIN_SCORE = 15, 200
assert sum(FULL_NUM) == 79
RSUP = 79  
WIN= WIN_SCORE; EXT=WIN+RSUP-1

#=================================================================
# Base strategy class (Une politique qui s'applique à toutes les stratégies algorithmiques)
# Comme la règle officielle de Flip 7 nous oblige à donner notre deuxième Seconde Chance,
# on décide de toujours l'offrir au joueur qui a le plus petit score (le moins menaçant).
class strategy:
    """Politique de DON par défaut = au plus petit score (le moins menaçant)."""
    def choose_sc_recipient(self, view, eligible):
        return min(eligible, key=lambda q: q.score)
#=================================================================
#Strategy EV optimal (Valeur d'espérance optimale d'une manche)
def make_dp(deck0, sc0):
    # DP exact SANS remise (référence). value(held, sc).
    deck0 = tuple(deck0); memo = {}
    def value(deck, sc_deck, held, sc):
        key = (deck, sc_deck, held, sc); r = memo.get(key)
        if r is not None: return r
        tot = sum(deck) + sc_deck; v = sum(held)
        if tot == 0: memo[key] = float(v); return float(v)
        full = (len(held) == 6); hit = 0.0
        for n in range(13):
            cn = deck[n]
            if cn == 0: continue
            p = cn / tot
            if n in held:
                if sc:
                    nd = list(deck); nd[n] -= 1
                    hit += p * value(tuple(nd), sc_deck, held, 0)
                # sinon bust -> 0
            else:
                nd = list(deck); nd[n] -= 1
                if full: hit += p * (v + n + FLIP7_BONUS)
                else:    hit += p * value(tuple(nd), sc_deck, tuple(sorted(held + (n,))), sc)
        if sc_deck:
            hit += (sc_deck / tot) * value(deck, sc_deck - 1, held, (sc or 1))
        res = max(float(v), hit); memo[key] = res; return res
    return lambda held, sc: value(deck0, sc0, tuple(sorted(held)), sc)

def myopic_hit(held_set, sc, deck, sc_cnt):
    # Règle myope O(13) : tirer ssi E[gain à 1 coup] > 0.
    total = sum(deck) + sc_cnt
    if total == 0: return False
    dup = sum(deck[m] for m in held_set)
    if dup == 0 or sc: return True                        # aucun bust possible / SC en réserve
    inv = 1.0 / total
    gain = sum(deck[n] * inv * n for n in NUMBERS if n not in held_set)
    risk = dup * inv * sum(held_set)
    return gain > risk

def optimal_hit(held_set, sc, deck, sc_cnt):
    # Décision EV-optimale (hybride myope + DP).
    if myopic_hit(held_set, sc, deck, sc_cnt): return True
    held = tuple(sorted(held_set)); v = sum(held)
    return make_dp(tuple(deck), sc_cnt)(held, sc) > v + 1e-9

# Myopic strategy class
class Myopic(strategy): 
    def decide_hit(self, gs): 
        return myopic_hit(gs.me.held, gs.me.sc, gs.deck, gs.sc)

# Optimal strategy class
class Optimal(strategy): 
    def decide_hit(self, gs): 
        return optimal_hit(gs.me.held, gs.me.sc, gs.deck, gs.sc)

#=================================================================
# Classe pour l'objectif réel (Stratégie optimale pour maximiser la probabilité de gagner)

def gp_curve(p2, p3, Wv, f2, f3):
    """
    Calcule la courbe d'espérance de gain global G[s] pour chaque score cible 's' 
    atteignable par le joueur 1 à la fin de sa propre manche.
    
    Formule mathématique évaluée : E_{x2,x3}[Psi_1(s, p2+x2, p3+x3)]
    
    Arguments:
        p2, p3 : Scores cumulés actuels des deux adversaires.
        Wv     : Tenseur V(p1, p2, p3) de la probabilité de victoire à long terme.
        f2, f3 : Vecteurs de distribution de probabilité (PMF) des scores des adversaires sur la manche.
    """
    K=len(f2); x2=np.arange(K); x3=np.arange(K)

    # Broadcasting pour évaluer toutes les combinaisons de scores de fin de manche des adversaires
    bb=p2+x2[:,None] 
    cc=p3+x3[None,:] 
    fw=f2[:,None]*f3[None,:]

    G=np.empty(EXT+1)
    for s in range(EXT+1):
        mx=np.maximum(np.maximum(bb,cc),s) 
        ended=mx>=WIN

        # Gestion des égalités au sommet (partage équitable de la victoire si s est max)
        is1=(s==mx) 
        k=is1.astype(float)+(bb==mx)+(cc==mx)
        term=np.where(is1,1.0/np.maximum(k,1.0),0.0)
        bbi=np.clip(bb,0,WIN-1)
        cci=np.clip(cc,0,WIN-1)
        
        # Si la partie ne se termine pas ce coup-ci, on projette sur l'état de la manche suivante via Wv
        cont=Wv[min(s,WIN-1),bbi,cci] if s<WIN else np.zeros_like(fw)
        G[s]=float((fw*np.where(ended,term,cont)).sum())
    return G

def make_dp_winprob(deck, sc_cnt, G, p1):
    num_total=sum(deck); present=[n for n in NUMBERS if deck[n]>0]; memo={}
    def gval(b): return G[min(p1+b,EXT)]
    def value(held,sc,scc):
        key=(held,sc,scc); c=memo.get(key)
        if c is not None: return c
        v=sum(held); full=(len(held)==6); tot=num_total+scc
        if tot==0: memo[key]=gval(v); return gval(v)
        inv=1.0/tot; hit=0.0; dup=0
        for n in present:
            cn=deck[n]
            if n in held: dup+=cn; continue
            p=cn*inv
            hit+= p*gval(v+n+FLIP7_BONUS) if full else p*value(tuple(sorted(held+(n,))),sc,scc)
        if dup:
            if sc: hit+=(dup*inv)*value(held,0,scc)        # SC absorbe
            else:  hit+=(dup*inv)*gval(0)                  # bust -> banque 0, on continue depuis p1
        if scc: hit+=(scc*inv)*value(held,(sc or 1),scc-1)
        res=max(gval(v),hit); memo[key]=res; return res
    return value, gval

def winprob_hit(held_set, sc, deck, sc_cnt, G, p1):
    total=sum(deck)+sc_cnt
    if total==0: return False
    dup=sum(deck[m] for m in held_set)
    if dup==0 or sc: return True                            # raccourcis valides (G monotone)
    value,gval=make_dp_winprob(tuple(deck),sc_cnt,G,p1)
    held=tuple(sorted(held_set)); v=sum(held); stop=gval(v)
    num_total=sum(deck); inv=1.0/(num_total+sc_cnt); full=(len(held)==6); hit=0.0
    for n in NUMBERS:
        if deck[n]==0 or n in held: continue
        p=deck[n]*inv
        hit+= p*gval(v+n+FLIP7_BONUS) if full else p*value(tuple(sorted(held+(n,))),sc,sc_cnt)
    hit+=(dup*inv)*gval(0)
    if sc_cnt: hit+=(sc_cnt*inv)*value(held,1,sc_cnt-1)
    return hit>stop+1e-12

class WinProb(strategy):
    # Arrêt optimal sous P(win) : EV-optimal partout sauf couche de seuil, où w -> G_p.
    def __init__(self, W, fX): self.W=W; self.fX=fX; self._cache={}
    def decide_hit(self, gs):
        opp=[p.score for p in gs.players if p is not gs.me]
        p2,p3=(opp+[0,0])[:2]; key=(min(p2,199),min(p3,199))
        G=self._cache.get(key)
        if G is None: G=gp_curve(key[0],key[1],self.W,self.fX,self.fX); self._cache[key]=G
        return winprob_hit(gs.me.held, gs.me.sc, gs.deck, gs.sc, G, min(gs.me.score,199))

#=================================================================
# Class pour strategy humaine (Strategie interactive)

def _format_hand(held, sc):
    cards = " ".join(str(v) for v in sorted(held)) if held else "(vide)"
    return f"[{cards}]" + (" +SC" if sc else "")


def _format_deck(deck, sc_left):
    parts = [f"{v}:{deck[v]}" for v in range(13) if deck[v] > 0]
    return " ".join(parts) + f"  |  SC restantes: {sc_left}"

class Human:
    """Stratégie interactive : montre l'état réel et demande la décision au clavier."""

    def __init__(self, name="Toi", show_hint=True):
        self.name = name
        self.show_hint = show_hint

    def decide_hit(self, gs):
        me = gs.me
        print(f"\n--- {self.name}, à toi de jouer ---")
        print(f"Ta main          : {_format_hand(me.held, me.sc)}  "
              f"(banque si arrêt : {me.points()} pts)")
        print(f"Ton score total  : {me.score}")
        for o in gs.opp():
            print(f"  {o.name:10s} score={o.score:3d}  main={_format_hand(o.held, o.sc)}")
        print(f"Deck restant     : {_format_deck(gs.deck, gs.sc)}")
        if self.show_hint:
            total = sum(gs.deck) + gs.sc
            if total:
                dup = sum(gs.deck[v] for v in me.held)
                print(f"  (indice : risque de bust au prochain tirage = {dup/total:.0%})")
        while True:
            rep = input("Tirer ? (o/n) : ").strip().lower()
            if rep in ("o", "oui", "y", "yes"):
                return True
            if rep in ("n", "non", "no"):
                return False
            print("Réponds par 'o' ou 'n'.")

    def choose_sc_recipient(self, gs, eligible):
        print(f"\n{self.name}, tu as une Seconde Chance en trop — à qui la donner ?")
        for i, q in enumerate(eligible):
            print(f"  [{i}] {q.name} (score {q.score})")
        while True:
            rep = input("Choix : ").strip()
            if rep.isdigit() and 0 <= int(rep) < len(eligible):
                return eligible[int(rep)]
            print("Choix invalide.")
#=================================================================
class Announce:
    """Enveloppe une stratégie existante pour afficher sa décision, sans toucher à sa logique."""

    def __init__(self, name, inner):
        self.name = name
        self.inner = inner

    def decide_hit(self, gs):
        result = self.inner.decide_hit(gs)
        me = gs.me
        action = "tire" if result else "s'arrête"
        print(f"{self.name:10s} {action:9s}  main={_format_hand(me.held, me.sc)}  score={me.score}")
        return result

    def choose_sc_recipient(self, gs, eligible):
        chooser = getattr(self.inner, "choose_sc_recipient", None)
        recipient = chooser(gs, eligible) if chooser else min(eligible, key=lambda q: q.score)
        print(f"{self.name} donne sa Seconde Chance en trop à {recipient.name}.")
        return recipient
#=================================================================
#  Class pour strategy seuil
class Threshold(strategy): 
    def __init__(self, T=25): 
        self.T = T
    def decide_hit(self, gs): 
        return sum(gs.me.held) < self.T

