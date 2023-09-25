from dataclasses import dataclass
from tokenize import String
import numpy as np
from typing import Tuple, List
from itertools import tee, chain
from datetime import timedelta, datetime
from random import choice
from math import sqrt, log
import string

# Constants
CALC_SECS = 1
EXPLORE = sqrt(2)
PENALTY = 10

@dataclass(order=True)
class Constraint:
    ev: float
    p: float
    size: int
    i: int
    s: str

class Game:
    def start(self, cards):
        return "Z" * 24 + (frozenset(cards), 0)

    def next_state(self, state, play):
        is_npc = state[-1]
        state = list(state)
        h, a = play
        slot = (h%12)
        state[slot if state[slot]=="Z" else slot+12] = a 
        if not is_npc:
            state[24] -= {a}
        state[-1] = (state[-1] + 1) % 3

        return tuple(state)

    def legal_plays(self, state):
        is_npc = state[-1]
        hand = state[24]
        np_cards = set(string.ascii_uppercase) - (set(state[:24]) | hand | {"Y"})
        hrs_playable = {i%12 for i in range(24) if state[i] == "Z"}

        plays = [
            (h, a)
            for h in hrs_playable
            for a in (np_cards if is_npc else hand)
        ]

        return plays

    def is_over(self, state):

        return "Z" not in state[:24]

    def score(self, state, g, constraints):
        pairs = set()
        for i in range(24):
            for j in range(24):
                if g[i][j]:
                    pairs.add((chr(ord("A") + i), chr(ord("A") + j)))

        for i, a in enumerate(state[:24]):
            for j in chain(range(1, 5), range(13, 18)):
                b = state[(i+j)%24]
                pairs.discard((a, b))

        # itertools.pairwise() in python 3.10
        def pairwise(iterable):
            # pairwise('ABCDEFG') --> AB BC CD DE EF FG
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        score = 0
        for c in constraints:
            if any(p in pairs for p in pairwise(c.s)):
                score -= PENALTY
            else:
                n = c.size
                score += 1 if n == 2 else 3.0*2**(n-3)

        return score

class Player:
    def __init__(self, rng: np.random.Generator) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
            golf_map (sympy.Polygon): Golf Map polygon
            start (sympy.geometry.Point2D): Start location
            target (sympy.geometry.Point2D): Target location
            map_path (str): File path to map
            precomp_dir (str): Directory path to store/load precomputation
        """
        self.rng = rng

        self.game = Game()
        self.visits = {}
        self.vals = {}

        self.calc_time = timedelta(seconds=CALC_SECS)
        self.C = EXPLORE

        self.graph = None
        self.constraints = []

    def __calc_ev(self, p, size):

        return 2.0*p-1.0 if size==2 else p-1+p*3.0*2**(size-3)

    def __approx_p(self, cards, constraint):
        p = 1.0
        n = len(constraint)

        for i in range(1, n):
            if i < n-1 and constraint[i] in cards:
                if (constraint[i-1] not in cards) and (constraint[i+1] not in cards):
                    p *= 0.94
            if (constraint[i-1] not in cards) and (constraint[i] not in cards):
                p *= 10/23
            else:
                p *= 0.98

        return p

    #def choose_discard(self, cards: list[str], constraints: list[str]):
    def choose_discard(self, cards, constraints):
        """Function in which we choose which cards to discard, and it also inititalises the cards dealt to the player at the game beginning

        Args:
            cards(list): A list of letters you have been given at the beginning of the game.
            constraints(list(str)): The total constraints assigned to the given player in the format ["A<B<V","S<D","F<G<A"].

        Returns:
            list[int]: Return the list of constraint cards that you wish to keep. (can look at the default player logic to understand.)
        """
        g = [[False for _ in range(24)] for _ in range(24)]
        q, ret = [], []

        for i, c in enumerate(constraints):
            s = c.replace('<', '')
            size = len(s)
            p = self.__approx_p(cards, s)
            ev = self.__calc_ev(p, size)
            if ev > 0:
                q.append(Constraint(ev, p, size, i, s))

        # itertools.pairwise() in python 3.10
        def pairwise(iterable):
            # pairwise('ABCDEFG') --> AB BC CD DE EF FG
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        deps = set(cards)
        dcnt = 0
        while (q):
            q = sorted(q) 
            c = q.pop()
            ret.append(constraints[c.i])
            self.constraints.append(c)

            added = set()
            for a, b in pairwise(c.s):
                if not a in deps:
                    deps.add(a)
                    dcnt += 1
                if not b in deps:
                    deps.add(b)
                    dcnt += 1
                if not g[ord(a) - ord('A')][ord(b) - ord('A')]:
                    g[ord(a) - ord('A')][ord(b) - ord('A')] = True
                    added.add((a,b))

            # scale P(constraint) and recalc its ev
            for constraint in q:
                s = constraint.s
                for x, y in pairwise(s):
                    if not g[ord(x) - ord('A')][ord(y) - ord('A')]:
                        for a, b in added:
                            if (x, y) == (a, b):
                                continue
                            if x == a:
                                cnt = sum(g[ord(a) - ord('A')])
                                constraint.p *= (10 - cnt) / (11 - cnt)
                            if y == b:
                                cnt = sum(row[ord(b) - ord('A')] for row in g)
                                constraint.p *= (10 - cnt) / (11 - cnt)

                for _ in range(dcnt):
                    n = sum(e not in deps for e in s)
                    d = len(deps)
                    constraint.p *= (24 - d - n) / (25 - d - n)

                constraint.ev = self.__calc_ev(constraint.p, constraint.size)
                
            dcnt = 0
            q = list(filter(lambda c: c.ev > 0, q))

        self.graph = g

        return ret

    def __prune(self, hand, states):
        constraints = self.constraints
        evs = []
        for a in hand:
            ev = sum(c.ev for c in constraints if a in c.s)
            evs.append((ev, a))

        discard = sorted(evs)[0][1]
        pruned = [(p, s) for p, s in states if p[1] == discard]

        return pruned

    def __simulate(self, state):
        visits, vals, C = self.visits, self.vals, self.C
        visited = set()
        
        if state not in visits:
            visits[state] = 0
            vals[state] = 0

        expand = True
        while not (self.game.is_over(state)):
            # select
            legal = self.game.legal_plays(state)
            next_states = [(p, self.game.next_state(state, p)) for p in legal]

            is_npc = state[-1]
            hand = state[24]
            # prune tree if t < turn 3
            if not is_npc and len(hand) > 6:
                next_states = self.__prune(hand, next_states)

            if not is_npc and all((s in visits) for p, s in next_states):
                logN = log(max(visits[state], 1))

                _, move, state = max(
                    ((vals[s] / visits[s]) +
                     C * sqrt(logN / visits[s]), p, s)
                    for p, s in next_states
                )
            else:
                no_visits = [(p, s) for p, s in next_states if s not in visits]
                move, state = choice(no_visits if len(no_visits) else next_states)

            # expand
            if expand and state not in self.visits:
                expand = False
                visits[state] = 0
                vals[state] = 0
            
            visited.add(state)

        score = self.game.score(state, self.graph, self.constraints)

        # backpropagate
        for s in visited:
            if s not in visits:
                continue
            visits[s] += 1
            vals[s] += score
                
    def __encode_state(self, state, cards):
        hand = frozenset(cards) - frozenset(state)

        return tuple(state) + (frozenset(hand), 0)

    #def play(self, cards: list[str], constraints: list[str], state: list[str], territory: list[int]) -> Tuple[int, str]:
    def play(self, cards, constraints, state, territory):
        """Function which based n current game state returns the distance and angle, the shot must be played

        Args:
            score (int): Your total score including current turn
            cards (list): A list of letters you have been given at the beginning of the game
            state (list(list)): The current letters at every hour of the 24 hour clock
            territory (list(int)): The current occupiers of every slot in the 24 hour clock. 1,2,3 for players 1,2 and 3. 4 if position unoccupied.
            constraints(list(str)): The constraints assigned to the given player

        Returns:
            Tuple[int, str]: Return a tuple of slot from 1-12 and letter to be played at that slot
        """
        #Do we want intermediate scores also available? Confirm pls

        state = self.__encode_state(state, cards)
        legal = self.game.legal_plays(state)

        if not legal:
            return 
        if len(legal) == 1:
            return legal[0]

        games = 0
        begin = datetime.utcnow()
        while datetime.utcnow() - begin < self.calc_time:
            self.__simulate(state)
            games += 1

        next_states = [(p, self.game.next_state(state, p)) for p in legal]

        print(f"Simulated {games} games")

        print(f"Current state: {state}")
        print(f"Next moves:")
        for p, s in next_states:
            print(f"{p}: {self.visits.get(s)}, {self.vals.get(s, 0) / self.visits.get(s, 1):.2f}")

        
        mu_v, p =  max(
            (self.vals.get(s, 0) / self.visits.get(s, 1),
             p)
            for p, s in next_states
        )

        print(f"Played {p}: {mu_v:.2f}")

        h, a = p
        if not h:
            h = 12

        return h, a
