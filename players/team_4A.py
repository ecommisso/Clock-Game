from dataclasses import dataclass
from tokenize import String
import numpy as np
import numpy.typing as npt
import random
import string
from typing import Tuple, List


@dataclass(order=True)
class Constraint:
    ev: float
    p: float
    i: int
    s: str

@dataclass
class Node:
    state: npt.ArrayLike
    hour: int
    letter: str
    score: int = 0
    N: int = 0


class Tree:
    def __init__(self, root: "Node"):
        self.root = root
        self.root.children = []
        self.nodes = {root.state.tobytes(): root}

    def add(self, node: "Node"):
        self.nodes[node.state.tobytes()] = node
        self.root.children.append(node)

    def get(self, state: npt.ArrayLike):
        flat_state = state.tobytes()
        if flat_state not in self.nodes:
            return None
        return self.nodes[flat_state]


class Player:
    def __init__(self, rng: np.random.Generator) -> None:
        """Initialise the player with given skill.

        Args:
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
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

    # https://cs.stackexchange.com/questions/106539/how-to-find-the-best-exploration-parameter-in-a-monte-carlo-tree-search
    def __score(self, constraints: list[str], final_state: list[str]):
        """
        helper function for calculating the utility
        """
        score_value_list = [
            1, 3, 6, 12]  # points for satisfying constraints on different lengths
        score = 0
        for i in range(len(constraints)):
            list_of_letters = constraints[i].split("<")
            constraint_true_indic = True
            for j in range(len(list_of_letters)-1):
                distance_difference = (final_state.index(
                    list_of_letters[j+1]) % 12) - (final_state.index(list_of_letters[j]) % 12)
                if distance_difference < 0:
                    distance_difference = distance_difference + 12
                if not (distance_difference <= 5 and distance_difference > 0):
                    constraint_true_indic = False
                if constraint_true_indic == False:
                    score = score - 1
                else:
                    score = score + score_value_list[len(list_of_letters) - 2]
        return score

    def __utility(self, constraints: list[str], final_state: list[str]):
        """Utility function that returns player's score after a single monte carlo simulation

        Args:
            final_state (list(str)): The simulated letters at every hour of the 24 hour clock
            constraints(list(str)): The constraints assigned to the given player

        Returns:
            int: player's core after a single monte carlo simulation
        """
        myscore = self.__score(constraints, final_state)

        for opponent in range(2):
            oppo_cons = list()
            for idx in range(len(constraints)): # create some random constraints for opponents        
                oppo_cons.append("<".join(self.rng.choice(list(string.ascii_uppercase)[:24], 2+(idx%4), replace = False)))
            if myscore < self.__score(oppo_cons, final_state): # beaten by a player
                return 0
        return 1

    def __select(self, tree: "Tree", state: list[str], alpha: float = np.sqrt(2)):
        """Starting from state, move to child node with the
        highest UCT value.

        Args:
            tree ("Tree"): the search tree
            state (list[str]): the clock game state
            alpha (float): exploration parameter [PERHAPS THIS CAN BE DETERMINED IN RISKY_VS_SAFE()?]
        Returns:
            state: the clock game state after best UCT move
        """

        max_UCT = 0.0
        move = state

        for child_node in tree.root.children:
            node_UCT = (child_node.score/child_node.N + alpha *
                        np.sqrt(tree.root.N/child_node.N))
            if node_UCT > max_UCT:
                max_UCT = node_UCT
                move = child_node

        return move

    def __expand(self, tree: "Tree", cards: list[str], state: list[str]):
        """Add all children nodes of state into the tree and return
        tree.

        Args:
            tree ("Tree"): the search tree
            cards (list[str]): cards from our player
            state (list[str]): the clock game state
        Returns:
            "Tree": the tree after insertion
        """

        for letter in cards:
            # add our letters in every hour available
            for i in range(0, 12):
                new_state = np.copy(state)
                if new_state[i] == 'Z':
                    new_state[i] = letter
                elif new_state[i+12] == 'Z':
                    # if hour already occupied, try index + 12
                    new_state[i+12] = letter
                else:
                    # if both slots of hour already occupied, continue
                    continue
                hour = 12 if i == 0 else i
                tree.add(Node(np.array(new_state), hour, letter, 0, 1))
        return tree

    def __simulate(self, tree: "Tree", state: npt.ArrayLike, constraints: list[str], remaining_cards: list[str]):
        """Run one game rollout from state to a terminal state using random
        playout policy and return the numerical utility of the result.

        Args:
            tree ("Tree"): the search tree
            state (list[str]): the clock game state
            constraints (list[str]): constraints our player wants to satisfy
            remaining_cards (list[str]): cards from all players not yet played

        Returns:
            "Tree": the search tree with updated scores
        """
        new_state = np.copy(state)
        while len(remaining_cards):
            rand_letter = remaining_cards.pop(
                random.randint(0, len(remaining_cards) - 1))
            available_hours = np.where(new_state == 'Z')
            hour = random.choice(available_hours[0])
            new_state[hour] = rand_letter

        score = self.__utility(constraints, new_state.tolist())
        cur_node = tree.get(state)
        cur_node.score += score
        cur_node.N += 1
        tree.root.score += score
        tree.root.N += 1

        return tree

    def __MCTS(self, cards: list[str], constraints: list[str], state: list[str], rollouts: int = 1000):
        # MCTS main loop: Execute MCTS steps rollouts number of times
        # Then return successor with highest number of rollouts
        tree = Tree(Node(np.array(state), 24, 'Z', 0, 1))
        tree = self.__expand(tree, cards, state)
        possible_letters = list(string.ascii_uppercase)[:24]
        for letter in state:
            if letter != 'Z':
                possible_letters.remove(letter)

        for i in range(rollouts):
            available_letters = possible_letters.copy()
            move = self.__select(tree, state)
            available_letters.remove(move.letter)
            tree = self.__simulate(
                tree, move.state, constraints, available_letters)

        nxt = None
        plays = 0

        for succ in tree.root.children:
            if succ.N > plays:
                plays = succ.N
                nxt = succ
        return nxt

    # def play(self, cards: list[str], constraints: list[str], state: list[str], territory: list[int]) -> Tuple[int, str]:

    def play(self, cards, constraints, state, territory):
        """Function which based n current game state returns the distance and angle, the shot must be played

        Args:
            score (int): Your total score including current turn
            cards (list): A list of letters you have been given at the beginning of the game
            state (list[str]): The current letters at every hour of the 24 hour clock
            territory (list[int]): The current occupiers of every slot in the 24 hour clock. 1,2,3 for players 1,2 and 3. 4 if position unoccupied.
            constraints(list[str]): The constraints assigned to the given player

        Returns:
            Tuple(int, str): Return a tuple of slot from 1-12 and letter to be played at that slot
        """
        move = self.__MCTS(cards, constraints, state)
        return move.hour, move.letter
