import copy

import numpy as np
from tokenize import String
from typing import Tuple, List

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

        # Use this as a selection strategy during play making.
        self.curr_constraint_tally = None

    def constraint_parser(self, _constraint):
        """A function that converts a constraint in string format to a list format.

        Args:
            _constraint(string) : The input string in the format "A<B<C".

        Returns:
            list[str] : Return a string in the form of ["A", "B", "C"]
        """
        return _constraint.upper().split('<')

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

        
        # print("Constraints given to us:", constraints)
        # print("Cards given to us:", cards)


        def assign_score_to_const(_constraint, _cards):

            """Function to assign a score based on expected reward to risk involved in picking the constraint at the beggining of the game.

            Args:
                _constraint(string) : a string containing the current constraint for which the score needs to be computed.
                _cards(list) : A list of letters given at the beginning of the same.

            Returns:
                score(int) : A score assigned to the constraint based on its feasibility and score obtained if the constraint is satisfied.
            """

            # A dict keeping hardcoded probabilities for the three cases of two letter constraints
            two_letter_probs = {0: 0.43478260869, 1: 0.687, 2: 1} #this have been calculated and hard coded in 

            # Points associated with each constraint size
            points_for_consts = {2: 1, 3: 3, 4: 6, 5: 12}

            # print("Current constraint:", _constraint)
            lst_constraint = self.constraint_parser(_constraint)
            # print("List format constraint:", lst_constraint)

            #score = 1
            def num_letters(constraint:list, _cards: list): #count number of letters in constraint 
                num_letters = 0 
                for letter in constraint:
                    if letter in _cards:
                        num_letters +=1 
                return num_letters

            #calculate the number of letters you have in a constraint 
            num_letters = num_letters(lst_constraint, _cards) 
            #calculate expected probability of taking on that constraint 
            score = probability_calculator(num_letters, len(lst_constraint))
            '''print("constraint size: " + str(len(lst_constraint)))
            print("num letters: " + str(num_letters))
            print("score: " + str(score))'''
            #return the expected value of taking that constraint 
            return score * points_for_consts[len(lst_constraint)] - (1-score)

        if len(constraints) <= 3: #if 0 to 3 constraints 
            expected_value_lowerbound = .05
        
        elif len(constraints) <= 10: #if 4 to 10 constraints 
            expected_value_lowerbound = .1

        elif len(constraints) <= 25: #if 11 to 25 constraints 
            expected_value_lowerbound = .2
        else: #if more than 25 constraints, be very selective 
            expected_value_lowerbound = .35

        # loop over all the constraints
        for curr_const in constraints:
            curr_const_score = (assign_score_to_const(curr_const, cards))
            # print("Score assigned to current constraint is", curr_const_score)
            if curr_const_score  >= expected_value_lowerbound: #if this value is too high, it just chooses nothing  
                final_constraints.append(curr_const)

        # Look into conflicting constraints
        self.curr_constraint_tally = {const: False for const in final_constraints}

        # print("Final selected constraints:", final_constraints)
        return final_constraints

    def is_2_const_satisfied(self, _two_const, _state):
        """A Function that checks whether a two letter constraint is satisfied.

        Args:
            _two_const (tuple(str, str)): A tuple containing two letter constraint.
            _state (list(list)): The state of the clock.

        Returns:
            bool: A boolean value that descibes whether the two letter constraint is satisfied.
        """

        # print("\t\tCurrent 2 letter constraint to evaluate is `{}<{}'".format(_two_const[0], _two_const[1]))

        # Obtain the index of left and right elements of the constraint
        l_idx = _state.index(_two_const[0]) 
        r_idx = _state.index(_two_const[1])

        # print("\t\tleft and right indices", l_idx, r_idx)

        if ((l_idx // 2) + 7) % 12 > ((r_idx // 2) + 1) % 12:
            return True
        return False


    def is_constraint_satisfied(self, _card, _card_placement, _constraints, state, territory):
        """Function which checks if a card move with help improve the constraint condition.

        Args:
            _card (str): A string depicting which card is being considered to play currently.
            _card_placement (int): A number between 0-23 that describes the position of the card.
            state (list(list)): The current letters at every hour of the 24 hour clock
            territory (list(int)): The current occupiers of every slot in the 24 hour clock. 1,2,3 for players 1,2 and 3. 4 if position unoccupied.
            _constraints(str): The constraints selected before the start of the game

        Returns:
            int: Return an int that describes if a placement satisfies the constraint (even if partially). 1 - if constraint is helped by the move, 0 - neutral case, -1 - if the move is against the constraint
        """


        # print("Current move is to play card {} to position {}".format(_card, _card_placement))
        # print("_constraints", _constraints)

        # Obtain all the constraints that are affected by current card
        _rel_constraints = [_const for _const in _constraints if _card in _const]
        
        # print("_rel_constraints", _rel_constraints)
        
        # Return if no constraint is affected by this card
        if len(_rel_constraints) == 0:
            return 0
        
        _const_status = [True] * len(_rel_constraints)

        for idx, _const in enumerate(_rel_constraints):

            # print("\tLooking into constraint ", _const)
            lst_const = self.constraint_parser(_const)

            # Check if the contraint so far has been satisfied
            for i in range(len(lst_const) - 1):
                if lst_const[i] == _card and lst_const[i+1] in state:
                    # Check if placing the current card at its respective place will satisfy the "`_card'<`right_card'" constraint.
                    temp_state = copy.deepcopy(state)
                    temp_state[_card_placement] = _card
                    
                    # if constraint is not satisfied, then skip to the next constraint
                    if self.is_2_const_satisfied((_card, lst_const[i+1]), temp_state) is False:
                        _const_status[idx] = False
                        break
                
                elif lst_const[i] == _card and lst_const[i+1] not in state:
                    # if the next card is not present, just continue to the next index
                    # Look into this case later on!
                    continue
                        
                elif lst_const[i] in state and lst_const[i+1] == _card:
                    # Check if placing the current card at its respective place will satisfy the "`left_card'<`_card'" constraint.
                    temp_state = copy.deepcopy(state)
                    temp_state[_card_placement] = _card
                    
                    # if constraint is not satisfied, then skip to the next constraint
                    if self.is_2_const_satisfied((lst_const[i], _card), temp_state) is False:
                        _const_status[idx] = False
                        break

                elif lst_const[i] not in state and lst_const[i+1] == _card:
                    # if the next card is not present, just continue to the next index
                    # Look into this case later on!
                    continue
                
                elif lst_const[i] in state and lst_const[i+1] in state:
                    # if constraint is not satisfied, then skip to the next constraint
                    if self.is_2_const_satisfied((lst_const[i], lst_const[i+1]), state) is False:
                        _const_status[idx] = False
                        break

        # print("_const_status at the end", _const_status)

        if np.any(_const_status):
            return 1
        else:
            return -1


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
        # Do we want intermediate scores also available? Confirm pls
        
        # print("\n***\n")
        
        # print("State: ", state)
        # print("Territory: ", territory)
        # print("Current Constraint Tally", self.curr_constraint_tally)

        max_iter = 1000
        for i in range(max_iter):

            letter = self.rng.choice(cards)
            territory_array = np.array(territory)
            available_hours = np.where(territory_array == 4)
            hour = self.rng.choice(available_hours[0])          # because np.where returns a tuple containing the array, not the array itself

            # Check if current random play leads to any constraint's situation improving
            is_satisfied = self.is_constraint_satisfied(letter, hour, constraints, state, territory)
            if is_satisfied == 1:
                break
        
        # print("~~~~~~~~~~~Is Satisfied: ", is_satisfied, "~~~~~~~~~~~~~~~")

        hour = hour % 12 if hour % 12 != 0 else 12
        return hour, letter
def probability_calculator(num_letters, constraint_length):
    prob_yes_nol = 0.45
    prob_yes_onel = 0.687894539465445
    prob_yes_twol = 1
    #combine expected value depending on scores
    two_letter = prob_yes_onel*1 - (1-prob_yes_onel)
    #three_letter -> 3 different ones
    #num_letters = 1
    if constraint_length == 2:
        if num_letters == 0:
            probability = prob_yes_nol
        elif num_letters == 1:
            probability = prob_yes_onel
        else:
            probability = prob_yes_twol 
        
    elif constraint_length == 3:
        probability = prob_3letter(num_letters,prob_yes_nol,prob_yes_onel,prob_yes_twol)
    elif constraint_length == 4:
        probability = prob_4letter(num_letters,prob_yes_nol,prob_yes_onel,prob_yes_twol)
    #elif constraint_length == 5:
    else: #is 5 
        probability = prob_5letter(num_letters,prob_yes_nol,prob_yes_onel,prob_yes_twol)
    return probability 



def prob_3letter(num_letters,prob_yes_nol,prob_yes_onel,prob_yes_twol):
  if num_letters ==0:
    prob = (prob_yes_nol**3)
  elif num_letters == 1:
    #2 constraints of 1 letter, and 1 of 0
    prob = (prob_yes_onel**2)*(prob_yes_nol)
  elif num_letters == 2:
    #2 constraints of 1 letter, and 1 of 2
    prob = (prob_yes_onel**2)*(prob_yes_twol)
  else: #have all 3
    prob = 1
  return prob
#using number of letters you have -> combine the number
#if 1 letter -> 2
def prob_4letter(num_letters,prob_yes_nol,prob_yes_onel,prob_yes_twol):
  if num_letters ==0:
    prob = ((prob_yes_nol)**4)
  elif num_letters == 1:
    #3 constraints of 1 letter, and 3 of 0
    prob = ((prob_yes_onel)**3)*((prob_yes_nol)**3)
  elif num_letters == 2:
    #1 where you have both, 1 where you have none, 4 where you have 1
    prob = ((prob_yes_onel)**4)*(prob_yes_twol)*(prob_yes_nol)
  elif num_letters == 3:
    #3 where yo have both, 3 where you have one
    prob = ((prob_yes_onel)**3)*((prob_yes_twol)**3)
  else: #have all 4
    prob = 1 #as long as if you are pos 2/3 yo pick where 10 open slots remain #not exact cause up to 11 could be played before you go
  return prob
#five_letter
def prob_5letter(num_letters,prob_yes_nol,prob_yes_onel,prob_yes_twol):
  if num_letters ==0:
    prob = ((prob_yes_nol)**5)
  elif num_letters == 1:
    #4 constraints of 1 letter, and 6 of 0
    prob = ((prob_yes_onel)**4)*((prob_yes_nol)**6)
  elif num_letters == 2:
    #1 where you have both, 3 where you have none, 6 where you have 1
    prob = ((prob_yes_onel)**6)*(prob_yes_twol)*(prob_yes_nol**3)
  elif num_letters == 3:
    #3 where yo have both, 6 where you have one, 1 where you have none
    prob = ((prob_yes_onel)**6)*((prob_yes_twol)**3)*prob_yes_nol
  elif num_letters == 4:
    #4 where you have one, 6 where you have 2
    prob = ((prob_yes_onel)**4)*((prob_yes_twol)**6)
  else: #have all 5
    prob = 1 #as long as if you are pos 2/3 yo pick where 10 open slots remain #not exact cause up to 11 could be played before you go
  return prob