from tokenize import String
import numpy as np
from typing import Tuple, List
import random
import time
import string
import queue as Q

letters = list(string.ascii_uppercase)

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

    def choose_discard(self, cards, constraints):
        """Function in which we choose which cards to discard, and it also inititalises the cards dealt to the player at the game beginning

        Args:
            cards(list): A list of letters you have been given at the beginning of the game.
            constraints(list(str)): The total constraints assigned to the given player in the format ["A<B<V","S<D","F<G<A"].

        Returns:
            list[int]: Return the list of constraint cards that you wish to keep. (can look at the default player logic to understand.)
        """
        all_constraints = []
        constraint_list = [Q.PriorityQueue(), Q.PriorityQueue(), Q.PriorityQueue(), Q.PriorityQueue()]
        for i in range(len(constraints)):
            letter_list = constraints[i]
            all_letters = letter_list.split('<')
            length_of_letters = len(all_letters)
            missing_letters = len([letter for letter in all_letters if letter not in cards])
            # add constraints into priority queue based on expected value ranking
            if length_of_letters == 2:
                if missing_letters == 0:
                    constraint_list[length_of_letters - 2].put((10, letter_list))
                    all_constraints.append(letter_list)
                elif missing_letters == 1:
                    constraint_list[length_of_letters - 2].put((11, letter_list))
                    all_constraints.append(letter_list)
            elif length_of_letters == 3:
                if missing_letters == 2 and all_letters[1] in cards:
                    constraint_list[length_of_letters - 2].put((9, letter_list))
                    all_constraints.append(letter_list)
                elif missing_letters == 1:
                    constraint_list[length_of_letters - 2].put((8, letter_list))
                    all_constraints.append(letter_list)
                elif missing_letters == 0:
                    constraint_list[length_of_letters - 2].put((7, letter_list))
                    all_constraints.append(letter_list)
            elif length_of_letters == 4:
                if missing_letters == 2:
                    constraint_list[length_of_letters - 2].put((6, letter_list))
                    all_constraints.append(letter_list)
                elif missing_letters == 1:
                    constraint_list[length_of_letters - 2].put((5, letter_list))
                    all_constraints.append(letter_list)
                elif missing_letters == 0:
                    constraint_list[length_of_letters - 2].put((3, letter_list))
                    all_constraints.append(letter_list)
            elif length_of_letters == 5:
                if missing_letters == 2:
                    constraint_list[length_of_letters - 2].put((4, letter_list))
                    all_constraints.append(letter_list)
                elif missing_letters == 1:
                    constraint_list[length_of_letters - 2].put((2, letter_list))
                    all_constraints.append(letter_list)
                elif missing_letters == 0:
                    constraint_list[length_of_letters - 2].put((1, letter_list))
                    all_constraints.append(letter_list)

        final_tuples = []
        # Pick constraints - currently takes top 3 constraints of each constraint size
        for i in range(len(constraint_list)):
            # TODO: replace any dropped constraints from contradictions...
            # TODO: mess around with optimal number of constraints to keep per length, so not set to 1 for each
            for j in range(1):
                if not constraint_list[i].empty():
                    # count any singular contradiction as a contradiction
                    # TODO: Think of better way to handle contradictions based on groupings when C get large...
                    constraint_tuple = constraint_list[i].get()
                    if not self.check_contradiction(constraint_tuple, final_tuples):
                        final_tuples.append(constraint_tuple)
                else:
                    break
        final_constraints = [x[1] for x in final_tuples]
        return final_constraints

    # returns true if contradiction was found
    def check_contradiction(self, con_tuple, final_tuples):
        num_contradiction = 0
        for i in range(0, len(con_tuple[1]) - 2, 2):
            contradiction_substring = con_tuple[1][i:i + 3][::-1]
            old_con_list_size = len(final_tuples)
            # remove tuples with overlapping substring when new tuple has lower expected value ranking
            for j in range(old_con_list_size):
                if contradiction_substring in final_tuples[j][1]:
                    num_contradiction += 1

            # Skip the current constraint if multiple contradictions are found against it
            if num_contradiction > 1:
                return True

            final_tuples[:] = [x for x in final_tuples if not (contradiction_substring in x[1] and con_tuple[0] < x[0])]
            # if element contradiction present and removed, return false so that the new tuple gets added
            if num_contradiction == 1:
                return old_con_list_size == len(final_tuples)

        return False

    def play(self, cards, constraints, state, territory):
        """Determines the next move based on the current game state.

        Args:
            cards (list): A list of letters available to the player.
            constraints (list): The constraints assigned to the player.
            state (list): The current letters at every hour of the 24-hour clock.
            territory (list): The current occupiers of every slot in the 24-hour clock.

        Returns:
            tuple: A tuple containing the selected hour (1-12) and the letter to be played at that hour.

        This method initializes some variables and makes a preliminary check for two-letter constraints. If a valid move isn't found, it resorts to the maximize function to find the best possible move. It returns the selected move as a tuple containing the hour and the letter.
        """
        self.level = 0
        self.time = time.process_time()

        state_array = np.array(state).tolist()
        duplicate_territory = territory.copy()

        # Filter out infeasible constraints
        feasible_constraints = [c for c in constraints if self.is_constraint_feasible(c, state, cards)]

        # Use the maximize function directly
        child, util = self.maximize(cards, state_array, duplicate_territory, feasible_constraints, -10000, 10000)

        letter = child[0]
        hour = child[1]
        hour = hour % 12 if hour % 12 != 0 else 12

        return hour, letter

    def is_constraint_feasible(self, constraint, state, cards):
        """Check if a given constraint is feasible to satisfy with the current state and cards."""
        letters_in_constraint = constraint.split("<")
        missing_from_state = [letter for letter in letters_in_constraint if letter not in state]

        # If all missing letters are not in cards, then it's not feasible
        if all(letter not in cards for letter in missing_from_state):
            return False
        return True

    def minimize(self, cards, state, territory, constraints, alpha, beta):
        """... (existing docstring) ..."""

        self.level += 1
        curr_time = time.process_time() - self.time
        availableMoves = self.getAvailableMoves(cards, territory)
        available_hours = np.where(np.array(territory) == 4)[0]
        availableLetters = list(set(letters) - set(state) - set(cards))

        if len(available_hours) == 0 or curr_time >= 1:
            return [state, territory], self.getCurrentScore(constraints, state, territory)

        minChild, minUtil = None, 10000

        for letter in availableLetters:  # Loop through all available letters
            for i in range(2):
                for j in available_hours:
                    if self.rng.random() <= (1 / float(len(available_hours))):
                        state[j] = letter  # Place the available letter
                        territory[j] = 0 if i == 0 else 2

                        other, util = self.minimize(cards, state, territory, constraints, alpha, beta)
                        util += self.getCurrentScore(constraints, state, territory)

                        if util < minUtil:
                            minChild, minUtil = [letter, j], util

                        if minUtil <= alpha:
                            return minChild, minUtil

                        beta = min(beta, minUtil)

        return minChild, minUtil

    def maximize(self, cards, state, territory, constraints, alpha, beta):
        """Maximizes the utility value in the minimax algorithm with alpha-beta pruning.

        Args:
            cards (list): List of available cards to play.
            state (list): Current state of the board.
            territory (list): Current territory ownership status.
            constraints (list): List of constraints to satisfy.
            alpha (int): The current alpha value for alpha-beta pruning.
            beta (int): The current beta value for alpha-beta pruning.

        Returns:
            tuple: A tuple containing the best move and its utility value.

        This method represents the maximizing player in the minimax algorithm. It explores possible moves and evaluates them using the minimize method, attempting to maximize the utility value. Alpha-beta pruning is used to ignore branches with lower utility values.
        """
        self.level += 1
        curr_time = time.process_time() - self.time
        availableMoves = self.getAvailableMoves(cards, territory)

        if not availableMoves or curr_time >= 1:
            return [state, territory], self.getCurrentScore(constraints, state, territory)

        maxChild, maxUtil = None, -10000
        fallback_move = None

        for child in availableMoves:
            for currLocation in availableMoves[child]:
                if not fallback_move:
                    fallback_move = [child, currLocation]
                if self.is_move_valid(child, state, constraints, cards, currLocation):
                    state[currLocation] = child
                    territory[currLocation] = 1  # Current player

                    other, util = self.minimize(cards, state, territory, constraints, alpha, beta)
                    util += self.getCurrentScore(constraints, state, territory)

                    if util > maxUtil:
                        maxChild, maxUtil = [child, currLocation], util

                    if maxUtil >= beta:
                        return maxChild, maxUtil

                    alpha = max(alpha, maxUtil)

        if not maxChild and fallback_move:
            # Fallback to a random move if no valid moves are found
            maxChild = fallback_move

        return maxChild, maxUtil

    def is_move_valid(self, move, state, constraints, cards, hour):
        """Checks if a move is valid given the current constraints and state.

        Args:
            move (str): The letter representing the move to check.
            state (list): The current state of the board.
            constraints (list): List of constraints to satisfy.
            cards (list): List of available cards to play.
            hour (int): The hour slot where the move is intended to be played.

        Returns:
            bool: True if the move is valid, otherwise False.

        This method checks whether playing a specific card at a specific hour would satisfy the current constraints. It also checks additional conditions to prevent playing a card that would invalidate a constraint.
        """
        for constraint in constraints:
            if move in constraint:
                letters_in_constraint = constraint.split("<")
                if len(letters_in_constraint) == 2:
                    if letters_in_constraint.index(move) == 0 and letters_in_constraint[1] not in state and \
                            letters_in_constraint[1] not in cards:
                        return False
                    elif letters_in_constraint.index(move) == 1 and letters_in_constraint[0] not in state and \
                            letters_in_constraint[0] not in cards:
                        return False
                else:
                    if letters_in_constraint.index(move) > 0 and letters_in_constraint[
                        letters_in_constraint.index(move) - 1] not in state:
                        return False
                    # Additional check to prevent playing a card that would invalidate the constraint
                    prev_index = letters_in_constraint.index(move) - 1
                    next_index = letters_in_constraint.index(move) + 1
                    if prev_index >= 0 and letters_in_constraint[prev_index] in state:
                        prev_hour = state.index(letters_in_constraint[prev_index])
                        if not (0 < (hour - prev_hour) % 12 <= 5):
                            return False
                    if next_index < len(letters_in_constraint) and letters_in_constraint[next_index] in state:
                        next_hour = state.index(letters_in_constraint[next_index])
                        if not (0 < (next_hour - hour) % 12 <= 5):
                            return False
        return True

    def getAvailableMoves(self, cards, territory):
        """Gets the available moves given the current cards and territory.

        Args:
            cards (list): List of available cards to play.
            territory (list): Current territory ownership status.

        Returns:
            dict: A dictionary where keys are available cards and values are lists of available hours to play them.

        This method generates a dictionary of available moves given the current cards and territory status. Each card can be played at any available hour slot.
        """
        availableMoves = {}
        available_slots = np.where(np.array(territory) == 4)[0]

        for card in cards:
            availableMoves[card] = available_slots

        return availableMoves

    def getCurrentScore(self, constraints, state, territory):
        """Calculates the current score based on the constraints and the current state of the board.

        Args:
            constraints (list): List of constraints to satisfy.
            state (list): The current state of the board.
            territory (list): Current territory ownership status.

        Returns:
            int: The current score.

        This method calculates the current score by evaluating each constraint with the current state of the board. It adds points for satisfied constraints and deducts points for unsatisfied constraints, returning the total score.
        """
        letter_position = {}
        for i in range(len(state)):
            letter_position[state[i]] = i
        score = 0
        score_value_list = [1, 3, 12, 24]

        for constraint in constraints:
            list_of_letters = constraint.split("<")
            constraint_true_indic = True

            for i in range(len(list_of_letters) - 1):
                if list_of_letters[i + 1] in letter_position and list_of_letters[i] in letter_position:
                    distance_difference = (letter_position[list_of_letters[i + 1]] % 12) - (
                            letter_position[list_of_letters[i]] % 12)
                    if distance_difference < 0:
                        distance_difference += 12
                    if not (0 < distance_difference <= 5):
                        constraint_true_indic = False
                        break
                else:
                    constraint_true_indic = False
                    break

            if constraint_true_indic:
                score += score_value_list[len(list_of_letters) - 2]
            else:
                score -= 10

        return score * 10
