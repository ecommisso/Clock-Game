from tokenize import String
import numpy as np
from typing import Tuple, List
import time 

##########################################################
# Goals:   
# 1.) Fix the the minimax: See the logic behind choosing each play for each corresponding letter.
# 2.) Fix choosing constraints. After several runs, come up with an efficient method of choosing constraints.
# 3.) Fix the lag that occurs during game play.
# 4.) Attempt to setup the google cloud to implement several runs and store each score in an excel document.
# 5.) A way to check that a move will satisfy a constraint right in the current play 
##########################################################

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

        # print("FINAL")
        # print(final_constraints)
        return final_constraints

    def is_played(self, letter, state):
        for i in range(len(state)):
            if letter in state[i]:
                return (True, i)
        return (False, -1)
    
    # def bestMove (self, available_hours, cards):
    #     bestScore = float('-inf')
    #     moves = [[]]
    #     bestMove = []
    #     for i in available_hours:
    #         for x in cards:
    #             hour = i
    #             letter = x
    #             score = self.minimax(clock, hour, letter, 0, true)
    #             if (score>bestScore):
    #                 bestScore = score
    #                 bestMove = [i,x]
        
    #     return bestMove

        
    def getAvailableMoves(self, cards, state):
        availableMoves = {}
        state_array = np.array(state)
        available_hours = np.where(state_array == 'Z')
        for i in available_hours:
            for x in cards:
                availableMoves.append(i,x)
        return availableMoves
    

    # function to get unplayed cards on the board 
    def getOtherCards(self, cards, state):
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
        for i in cards:
            if i in letters:
                letters.remove(i)
        for i in state: 
            if i in letters: 
                letters.remove(i)
        return letters 

    # new minimax  
    # def minimax(self, state, cards, constraints, depth, is_maximizing):
    #     if depth == 8 or (time.process_time() - self.time) >= 1:
    #         score = self.getScore(state, cards, constraints)
    #         return None, score

    #     available_hours = np.where(np.array(state) == 'Z')[0]
    #     if is_maximizing:
    #         best_score = float('-inf')
    #         best_move = None

    #         for hour in available_hours:
    #             for card in cards:
    #                 if card not in state:
    #                     state[hour] = card, score = self.minimax(state, cards, constraints, depth + 1, False)
    #                     state[hour] = 'Z'

    #                     if score > best_score:
    #                         best_score = score
    #                         best_move = (hour, card)

    #         return best_move, best_score
    #     else:
    #         best_score = float('inf')
    #         best_move = None
    #         other_cards = self.getOtherCards(cards, state)

    #         for hour in available_hours:
    #             for card in other_cards:
    #                 if card not in state:
    #                     state[hour] = card, score = self.minimax(state, cards, constraints, depth + 1, True)
    #                     state[hour] = 'Z'

    #                     if score < best_score:
    #                         best_score = score
    #                         best_move = (hour, card)

    #         return best_move, best_score

    # our original minimax 
    def minimax(self, state, cards, constraints, depth, isMaximizing):
        bestMove = None
        score = self.getScore(state, cards, constraints)
        curr_time = time.process_time() - self.time
        # check what the score is/ who the "winner" is 
        if depth == 8 or curr_time >= 1:
            score = self.getScore(state, cards, constraints)
            return state, score

        state_array = np.array(state)
        available_hours = np.where(state_array == 'Z')
        # print("STATE:", state)
        # print("STATE ARR:", state_array)
        avail_arr = available_hours[0].flatten()
        # print("BEFORE", available_hours)
        # print("FLATTEN", avail_arr)
        # print("FLATTEN[0]: ", avail_arr[0])
        
        #maximizing
        if(isMaximizing):  
            bestScore = float('-inf')
            for i in avail_arr:
                for x in cards:
                    if x not in state:
                        state[i] = x
                        #print("STATE:", state)
                        #print("POS: ", i, "LETTER: ", x)
                        #we got rid of the plus sign here.....
                        score = self.minimax(state, cards, constraints, depth+1, False)[1]
                        state[i] = 'Z'
                        if (score>bestScore):
                            bestScore = score
                            bestMove = [i, x]
                            return bestMove, score

                '''
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
                '''

        #minimizing 
        #figure out how to minimize twice to represent the two players
        #add alpha beta
        else:
            bestScore = float('inf')
            other_cards = self.getOtherCards(cards, state)
            for i in avail_arr:
                for x in other_cards:
                    if x not in state:
                        state[i] = x
                        score = self.minimax(state, cards, constraints, depth+1, True)[1]
                        state[i] = 'Z'
                        if (score<bestScore):
                            bestScore = score
                            bestMove = [i, x]

        if bestMove is None:
            letter = self.rng.choice(cards)
            hour = self.rng.choice(avail_arr[0])          #because np.where returns a tuple containing the array, not the array itself
            hour = hour%12 if hour%12!=0 else 12
            bestMove =  hour, letter
                            
        return bestMove, bestScore
        

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

        # print("STATE: ", state)
        # print("CARDS: ", cards)

        self.time = time.process_time()
        new_cards = cards.copy() 
        new_state = state.copy()
        new_constraints = constraints.copy()
        depth = 0
        
        bestMove = self.minimax(new_state, new_cards, new_constraints, depth, True)[0]
        #print("check: ", bestMove)
        letter = bestMove[1]
        hour = bestMove[0]
        hour = hour%12 if hour%12!=0 else 12
        print("MOVE = ", hour, ", ", letter)
        # print("State", state)
        return hour, letter
    
    # func to get the score of our own cards at the passed in state 
    def getScore(self, state, cards, constraints):
        # if a letter satsifies anything += 1/X, X being the number of letters in that constraint
        totalScore = 0
        score_arr = [0, 0, 1, 3, 6, 12]
        positions = {}
        # create dict of key letters to index values 
        for i in range(len(state)):
            positions[state[i]] = i
         
        for i in range(len(constraints) - 1):
            constraint = constraints[i].split('<')
            for j in range(len(constraint) - 1):
                if constraint[j] in positions and constraint[j+1] in positions:
                    position1 = positions[constraint[j]]
                    position2 = positions[constraint[j+1]]
                    difference = (position2%12) - (position1%12)
                    if difference < 0:
                        difference += 12 
                    if difference <= 5:
                        totalScore += float(score_arr[(len(constraint))]/len(constraint))
                    else: 
                        totalScore -= 3
        
        #print("SCORE: ", totalScore)
        return totalScore    
