##########################################################
# Goals:   
# 1.) Fix the the minimax: See the logic behind choosing each play for each corresponding letter.
# 2.) Fix choosing constraints. After several runs, come up with an efficient method of choosing constraints.
# 3.) Fix the lag that occurs during game play.
# 4.) Attempt to setup the google cloud to implement several runs and store each score in an excel document.
# 5.) A way to check that a move will satisfy a constraint right in the current play 
##########################################################

import numpy as np
from typing import List, Tuple
import time 

class Player:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.time = 0

    #def choose_discard(self, cards: list[str], constraints: list[str]):
    def choose_discard(self, cards, constraints):
        """Function in which we choose which cards to discard, and it also inititalises the cards dealt to the player at the game beginning

        Args:
            cards(list): A list of letters you have been given at the beginning of the game.
            constraints(list(str)): The total constraints assigned to the given player in the format ["A<B<V","S<D","F<G<A"].

        Returns:
            list[int]: Return the list of constraint cards that you wish to keep. (can look at the default player logic to understand.)
        """
        
        final_constraints = []
        stripped_constraints = []
        # print("CONSTRAINTS")
        # print(constraints)

        # print("CARDS")
        # print(cards)

        for i in range(len(constraints)):
            # splits current constraint into array of letters and removes <'s 
            curr = constraints[i].split('<')
            while '<' in curr:
                curr.remove('<')
            numLetters = len(curr)
            
            # see how many letters in a constraint we have 
            matches = 0
            for letter in curr:
                if letter in cards:
                    matches += 1

            match numLetters:
                    case 2:
                        if(matches == 1) or (matches == 2):
                            final_constraints.append(constraints[i])
                    case 3:
                        if(matches == 2):
                            final_constraints.append(constraints[i])
                    case 4:
                        if(matches == 3):
                            final_constraints.append(constraints[i])
                        elif((curr[0] in cards and curr[2] in cards) or (curr[1] in cards and curr[3] in cards)):
                            final_constraints.append(constraints[i])
                    case 5:
                        if(matches == 4):
                            final_constraints.append(constraints[i])
                        elif((curr[0] in cards and curr[2] in cards and curr[4] in cards)):
                            final_constraints.append(constraints[i]) 

            # if we don't have any letters in a constraint don't choose it 

            #if(len(final_constraints) == 0){
            #}
  
            #if self.rng.random()<=0.5:
            #   final_constraints.append(constraints[i])

        print("FINAL")
        print(final_constraints)
        return final_constraints

    def is_played(self, letter: str, state: List[str]) -> Tuple[bool, int]:
        for i, hour in enumerate(state):
            if letter in hour:
                return True, i
        return False, -1

    def get_available_moves(self, cards: List[str], state: List[str]) -> List[Tuple[int, str]]:
        available_moves = []
        available_hours = [i for i, hour in enumerate(state) if 'Z' in hour]

        for i in available_hours:
            for x in cards:
                available_moves.append((i, x))

        return available_moves

    def get_other_cards(self, cards: List[str], state: List[str]) -> List[str]:
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
        return [letter for letter in letters if letter not in cards and all(letter not in hour for hour in state)]

    def minimax(self, state: List[str], cards: List[str], constraints: List[str], depth: int, is_maximizing: bool) -> Tuple[List, float]:
        best_move = None
        score = self.get_score(state, cards, constraints)
        curr_time = time.process_time() - self.time

        if curr_time >= 1:
            score = self.get_score(state, cards, constraints)
            return state, score

        available_hours = [i for i, hour in enumerate(state) if 'Z' in hour]

        if is_maximizing:  
            best_score = -1000
            for i in available_hours:
                for x in cards:
                    if x not in state[i]:
                        state[i] = x
                        for j in range(2):
                            score = self.minimax(state, cards, constraints, depth+1, False)[1]
                        state[i] = 'Z'
                        if score > best_score:
                            best_score = score
                            best_move = (i, x)
        else:
            best_score = 1000
            other_cards = self.get_other_cards(cards, state)
            for i in available_hours:
                for x in other_cards:
                    if x not in state[i]:
                        state[i] = x
                        score = self.minimax(state, cards, constraints, depth+1, True)[1]
                        state[i] = 'Z'
                        if score < best_score:
                            best_score = score
                            best_move = (i, x)

        if best_move is None:
            letter = self.rng.choice(cards)
            if not available_hours:
                # Handle the case where available_hours is empty (e.g., all hours are occupied)
                # You can choose a default action here, like choosing a random card and hour.
                # For example:
                hour = self.rng.choice(range(1, 13))  # Choose a random hour from 1 to 12
            else:
                hour = self.rng.choice(available_hours) % 12 or 12
            best_move = hour, letter

        return best_move, best_score

    def play(self, cards: List[str], constraints: List[str], state: List[str], territory: List[int]) -> Tuple[int, str]:
        self.time = time.process_time()
        new_cards = cards.copy() 
        new_state = state.copy()
        new_constraints = constraints.copy()
        depth = 0

        best_move, bestScore = self.minimax(new_state, new_cards, new_constraints, depth, True)
        letter = best_move[1]
        hour = best_move[0] % 12 or 12
        print("MOVE = ", hour, ", ", letter)
        print("score:", bestScore)
        return hour, letter

    def get_score(self, state: List[str], cards: List[str], constraints: List[str]) -> float:
        total_score = 0
        score_arr = [0, 0, 1, 3, 6, 12]
        positions = {}

        for i, hour in enumerate(state):
            for j, letter in enumerate(hour):
                positions[letter] = i

        #print("POSITIONS")
        #print(positions)

        for constraint in constraints:
            curr = constraint.split('<')
            curr = [c for c in curr if c.isalpha()]

            for i in range(len(curr) - 1):
                if curr[i] in positions and curr[i+1] in positions:
                    position1 = positions[curr[i]]
                    position2 = positions[curr[i+1]]
                    difference = (position2 % 12) - (position1 % 12)

                    if difference < 0:
                        difference += 12

                    if difference <= 5:
                        total_score += float(score_arr[len(curr)] / len(curr))
                    else:
                        total_score -= 3
        return total_score