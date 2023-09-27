from tokenize import String
import numpy as np
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
        self.discardPile = []
        self.queue = []
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

        for constraint in constraints: 
        #check every constraint to make sure we have atleast 1 letter in every pair in constraint
            if len(constraint) > 3: continue 
            if self.__checkPairs(cards, constraint): 
                final_constraints.append(constraint)

        self.__organizeCards(cards, final_constraints)
        return final_constraints



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
        letter = None  #because np.where returns a tuple containing the array, not the array itself
        hour = None          
        territory_array = np.array(territory)
        available_hours = np.where(territory_array == 4)
        #parse all constraints
        for constraint in constraints: 
            #if we have all letters then play this constraint
            if self.__haveAllLetters(cards, constraint): 
                letter = constraint[0]
                self.queue.append(constraint[2])
                break
            elif len(constraint) == 3: 
                #2 letter constraint where we have 1 letter, check that other letter was played 
                playedAt = self.__wasPlayedAt(constraint[2], state)
                if constraint[0] in cards and playedAt is not None: 
                    letter = constraint[0]
                    if letter in self.queue: self.queue.remove(letter)
                    hour = self.__chooseHour(playedAt, state, False)
                    break
                playedAt = self.__wasPlayedAt(constraint[0], state)
                if constraint[2] in cards and playedAt is not None: 
                    letter = constraint[2]
                    if letter in self.queue: self.queue.remove(letter)
                    hour = self.__chooseHour(playedAt, state, True)
                    break
        #play next in queue if not empty
        if letter is None and self.queue != []: 
            letter = self.queue.pop()
        #play from discard if not empty 
        elif letter is None: 
            if self.discardPile != []:
                letter = self.rng.choice(self.discardPile)
                self.discardPile.remove(letter)
            else: letter = self.rng.choice(cards)
        
        #territory_array = np.array(territory)
        #available_hours = np.where(territory_array == 4)
        if hour is None:   
            hour = self.rng.choice(available_hours[0])
            hour = hour%12 if hour%12!=0 else 12
        return hour, letter
    
    def __checkPairs(self, cards, constraint): 
    #treat constraint as collection of pairs and check each pair 
        i = 0
        while i < len(constraint)-1:  
            if not self.__havePair(cards, constraint[i], constraint[i+2]):
                return False
            i += 2
        return True

    def __havePair(self, cards, letter1, letter2): 
    #check if we have atleast 1 letter in pair of letters 
        return letter1 in cards or letter2 in cards
    
    #create discard pile 
    def __organizeCards(self, cards, constraints):
        for card in cards: 
            if self.__isDiscard(card, constraints): 
                self.discardPile.append(card)

    #determine if we have all letters in a constraint
    def __isDiscard(self, card, constraints): 
        for constraint in constraints: 
            if card in constraint: return False 
        return True 

    #check if we have all letters in constraint
    def __haveAllLetters(self, cards, constraint): 
        i = 0 
        while i < len(constraint):  
            if constraint[i] not in cards: 
                return False
            i+=2
        return True
    
    #check if letter was played
    def __wasPlayedAt(self, letter, state): 
        for hour, letterAtHour in enumerate(state): 
            if letter == letterAtHour: 
                return hour
        return None 
    
    #chooseHour that we want to play at when other letter in 2 letter constraint is played  
    def __chooseHour(self, hourPlayed, state, clockwise):
        if hourPlayed < 12: 
            hourPlayed = hourPlayed + 12 
        if clockwise:  
            i = 1
            while i < 6: 
                hour = (hourPlayed+i)%24
                if state[hour] == 'Z': return hour
                complimentary = self.__getComplimentary(hour)
                if state[complimentary] == 'Z': return complimentary
                i += 1 
        else: 
            i = 1 
            while i < 6: 
                hour = (hourPlayed-i)%24
                if state[hour] == 'Z': return hour 
                complimentary = self.__getComplimentary(hour)
                if state[complimentary] == 'Z': return complimentary
                i += 1


    def __getComplimentary(self, hour):
        if hour >= 12: 
            return hour%12
        else: 
            return hour+12

        



