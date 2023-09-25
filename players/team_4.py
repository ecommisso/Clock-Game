import copy
from dataclasses import dataclass
from tokenize import String
import numpy as np
import numpy.typing as npt
import random
import string
import time
from typing import Tuple, List
from itertools import tee


@dataclass
class Node:
    state: npt.ArrayLike
    parent: "Node"
    children: list["Node"]
    hour: int
    letter: str
    score: int = 0
    N: int = 0


class Tree:
    def __init__(self, root: "Node"):
        self.root = root
        self.nodes = {root.state.tobytes(): root}
        self.size = 1

    def add(self, node: "Node"):
        self.nodes[node.state.tobytes()] = node
        parent = node.parent
        parent.children.append(node)
        self.size += 1

    def get(self, state: list[str]):
        flat_state = state.tobytes()
        if flat_state not in self.nodes:
            return None
        return self.nodes[flat_state]


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

    """
    Team 8 Inspired MCTS for playing game
    """

    def is_constraint_valid(self, constraint: str, curr_state: list[str]):
        """Check if constraint is valid given current board state

        Args:
            constraint (str): The constraint to be checked (e.g. "A<B<C")
            curr_state (list[str]): The current board configuration

        Returns:
            bool: True if the constraint is valid. Else, false


        """

        for i in range(len(constraint) - 1):
            if constraint[i] not in curr_state or constraint[i+1] not in curr_state:
                return False

            position1 = curr_state.index(constraint[i])
            position2 = curr_state.index(constraint[i+1])
            distance = (position2 - position1) % 12

            if not (1 <= distance <= 5):
                return False

        return True

    def check_all_constraints(self, constraints: list[str], curr_state: list[str]):
        """Check if all constraints are valid given current board state

        Args:
            constraints (list[str]): The constraints to be checked (e.g. ["A<B<C", "S<D", "F<G<A"])
            curr_state (list[str]): The current board configuration

        Returns:
            bool: True if all constraints are valid. Else, false
        """

        for formatted_constraint in constraints:
            constraint = formatted_constraint.replace("<", "")
            if not self.is_constraint_valid(constraint, curr_state):
                return False

        return True


    def __utility(self, constraints: list[str], curr_state: list[str]):
        """Utility function that returns player's score after a single monte carlo simulation

        Args:
            final_state (list(str)): The simulated letters at every hour of the 24 hour clock
            constraints(list(str)): The constraints assigned to the given player

        Returns:
            int: player score after a single monte carlo simulation
        """
        score_map = {
            2: 1,
            3: 3,
            4: 6,
            5: 12,
        }
        score = 0
        for formatted_constraint in constraints:
            constraint = formatted_constraint.replace("<", "")

            if self.is_constraint_valid(constraint, curr_state):
                score += score_map[len(constraint)]
            else:
                score -= 1

        return score

    def __select(self, tree: "Tree", state: list[str], alpha: float):
        """Starting from state, find a terminal node or node with unexpanded
        children. If all children of a node are in tree, move to the one with the
        highest UCT value.

        Args:
            tree ("Tree"): the search tree
            state (list[str]): the game board state
            alpha (float): exploration parameter
        Returns:
            state: the game board state
        """

        max_UCT = -float("inf")
        max_state = tree.root

        for child_node in tree.root.children:
            if child_node.children:
                node_UCT = (child_node.score/child_node.N + alpha
                            * np.sqrt(np.log(tree.root.N)/child_node.N))
                if node_UCT > max_UCT:
                    max_UCT = node_UCT
                    max_state = child_node

        return max_state

    def __expand(self, tree: "Tree", cards: list[str], state: list[str], constraints: list[str]):
        """Add children nodes of state into the tree while respecting constraints.

        Args:
            tree ("Tree"): The search tree.
            cards (list[str]): Cards from our player.
            state (list[str]): The clock game state.
            constraints (list[str]): The constraints assigned to the player.

        Returns:
            "Tree": The tree after insertion.
        """
        curr_node = tree.root
        new_state = state.copy()

        # Shuffle the cards randomly
        random.shuffle(cards)

        for card in cards:
            # Shuffle the hours randomly
            available_hours = [i for i in range(24) if new_state[i] == 'Z']
            random.shuffle(available_hours)

            for hour in available_hours:
                new_state[hour] = card

                # Check if the new state satisfies any constraint
                for formatted_constraint in constraints:
                    constraint = formatted_constraint.replace("<", "")
                    # print("new state, hour", new_state, hour)

                    if self.is_constraint_valid(constraint, new_state):
                        new_node = Node(new_state, curr_node, [], hour, card)
                        print("NEW node: ", new_node)
                        tree.add(new_node)
                        return tree  # Successfully expanded

                # Reset state for the next iteration
                new_state[hour] = 'Z'

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
                self.rng.integers(0, len(remaining_cards)))
            rand_hour = self.rng.integers(0, 24)

            if new_state[rand_hour] == 'Z':
                new_state[rand_hour] = rand_letter
            else:
                continue

        score = self.__utility(constraints, new_state.tolist())

        # update scores of all nodes in path
        curr_node = tree.root
        while curr_node:
            curr_node.score += score
            curr_node.N += 1
            curr_node = curr_node.parent

        return tree

    def __MCTS(self, cards: list[str], constraints: list[str], state: list[str], time_limit=1.0):
        """Run MCTS from state for time_limit CPU seconds and return the best move.

        Args:
            cards (list[str]): Cards from our player, i.e. ["A", "B", "C", "D", "E", "F", "G", "H"].
            constraints (list[str]): The constraints assigned to the player, i.e. ["A<B<C", "S<D", "F<G<A"].
            state (list[str]): The clock game state, i.e. ["Z" for i in range(24)] initially
            time_limit (float): The maximum time (in CPU seconds) allowed for the MCTS search.

        Returns:
            Tuple[int, str]: Return a tuple of slot from 1-12 and letter to be played at that slot, i.e. (1, "A")
        """

        # initialize tree
        root_node = Node(np.array(state), None, [], 24, 'Z', 0, 1)
        tree = Tree(root_node)

        # initialize remaining cards
        remaining_cards = cards.copy()

        # initialize time limit
        start_time = time.process_time()
        rollouts = 0
        alpha = 1.0

        # run MCTS
        while time.process_time() - start_time < time_limit:
            selected_node = self.__select(tree, state, alpha)

            tree = self.__expand(tree, cards, selected_node.state, constraints)

            tree = self.__simulate(tree, selected_node.state, constraints, remaining_cards)

            # Update variables
            rollouts += 1
            # alpha = 1.0 / np.sqrt(rollouts)

        best_node = tree.root
        for child_node in tree.root.children:
            print("child node hour", child_node.hour)
            if child_node.score/child_node.N > best_node.score/best_node.N:
                best_node = child_node
        # best_node = max(tree.root.children, key=lambda x: x.score / x.N)

        return best_node




    # def play(self, cards: list[str], constraints: list[str], state: list[str], territory: list[int]) -> Tuple[int, str]:
    def play(self, cards, constraints, state, territory):
        """Function which based n current game state returns the distance and angle, the shot must be played

        Args:
            score (int): Your total score including current turn
            cards (list): A list of letters you have been given at the beginning of the game
            state (list[str]): The current letters at every hour of the 24 hour clock
            territory (list[int]): The current occupiers of every slot in the 24 hour clock. 1,2,3 for players 1,2 and 3. 4 if position unoccupied.
            constraints(list[str]): The constraints assigned to the given player (constraints in form "A<B<C")
            time_limit (float): The maximum time (in seconds) allowed for the MCTS search.

        Returns:
            Tuple(int, str): Return a tuple of slot from 1-12 and letter to be played at that slot
        """
        move = self.__MCTS(cards, constraints, state)
        print("PLAYER 4 MOVE: ", move.hour, move.letter)
        return move.hour, move.letter
