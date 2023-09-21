from dataclasses import dataclass
from tokenize import String
import numpy as np
from typing import Tuple, List
from itertools import tee

@dataclass(order=True)
class Constraint:
    ev: float
    p: float
    i: int
    s: str

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

    def calc_ev(self, p, constraint):
        n = len(constraint)

        return 2.0*p-1.0 if len(constraint)==2 else p-1+p*3.0*2**(len(constraint)-3)

    def approx_p(self, cards, constraint):
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
            p = self.approx_p(cards, c)
            ev = self.calc_ev(p, s)
            if ev > 0:
                q.append(Constraint(ev, p, i, s))

        # itertools.pairwise() in python 3.10
        def pairwise(iterable):
            # pairwise('ABCDEFG') --> AB BC CD DE EF FG
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        while (q):
            q = sorted(q) 
            c = q.pop()
            ret.append(constraints[c.i])

            added = set()
            for a, b in pairwise(c.s):
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

                constraint.ev = self.calc_ev(constraint.p, s)
                q = list(filter(lambda c: c.ev > 0, q))

        return ret

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

        letter = self.rng.choice(cards)
        territory_array = np.array(territory)
        available_hours = np.where(territory_array == 4)
        hour = self.rng.choice(available_hours[0])          #because np.where returns a tuple containing the array, not the array itself
        hour = hour%12 if hour%12!=0 else 12
        return hour, letter
