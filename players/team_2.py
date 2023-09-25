from tokenize import String
import numpy as np
from typing import Tuple, List
import itertools as it

class Player:

    def __init__(self, rng: np.random.Generator) -> None:
        """Initialise the player.

        Args:
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
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
        for constraint in constraints:
            constraint_letters = constraint.split('<')
            keep = True
            for i in range(len(constraint_letters) - 1):
                if (constraint_letters[i] not in cards and constraint_letters[i + 1] not in cards):
                    keep = False
            if keep:
                final_constraints.append(constraint)
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
        
        # Alias to more comprehensible variable names
        letters = cards
        clock_letters = state
        all_constraints = constraints

        # Preprocessing
        active_constraints, useful_letters, discard_letters = self.process_constraints(all_constraints, letters, clock_letters)
        active_constraint_reprs = [self._build_constraint_repr(constraint, useful_letters, clock_letters) for constraint in active_constraints]
        useful_letter_stats = [UsefulLetterStats(active_constraints, active_constraint_reprs, letter) for letter in useful_letters]
        useful_letter_stats.sort(key=lambda s: (s.lpc_pos, s.lpc_dist_to_edge, s.lpc_dist_to_played_letter, s.lpc_len))
        playable_letter_stats = [s for s in useful_letter_stats if len(s.constraint_tups) == len(s.playable_constraint_tups)]

        # Compute clock stats (only for empty cells)
        clock_spaces_after = dict()
        clock_spaces_before = dict()
        clock_density = dict()  # number of filled cells in a 3-cell radius
        for hour, slot in get_all_empty(clock_letters):
            clock_spaces_after[(hour, slot)] = 0
            clock_spaces_before[(hour, slot)] = 0
            clock_density[(hour, slot)] = 0
            for h, s in it.product(crange(hour + 1, hour + 6, 1, 12), [0, 1]):
                if is_empty(clock_letters, h, s):
                    clock_spaces_after[(hour, slot)] += 1
            for h, s in it.product(crange(hour - 1, hour - 6, -1, 12), [0, 1]):
                if is_empty(clock_letters, h, s):
                    clock_spaces_before[(hour, slot)] += 1
            for h, s in it.product(crange(hour - 3, hour + 4, 1, 12), [0, 1]):
                if is_empty(clock_letters, h, s):
                    clock_density[(hour, slot)] += 1

        # Choose a letter and hour to play
        if len(playable_letter_stats) > 0:
            return self.play_letter_optimally(playable_letter_stats[0], clock_letters, clock_spaces_after, clock_spaces_before, clock_density), playable_letter_stats[0].letter
        else:
            return self.make_backup_play(active_constraints, useful_letters, discard_letters, clock_letters)


    def process_constraints(self, all_constraints, owned_letters, clock_letters):  # returns active_constraints, useful_letters, discard_letters
        active_constraints = [c.split('<') for c in all_constraints]
        active_constraints = [c for c in active_constraints if self._is_active_constraint(c, clock_letters)]
        active_constraints.sort(key=len, reverse=True)

        active_constraint_letters = sum(active_constraints, [])  # hacky way to flatten a 2D list
        useful_letters = list(set(owned_letters) & set(active_constraint_letters))
        discard_letters = list(set(owned_letters) - set(active_constraint_letters))
        
        return active_constraints, useful_letters, discard_letters
    

    def _is_active_constraint(self, constraint, clock_letters):
        completed = True
        for i in range(len(constraint) - 1):
            if constraint[i] in clock_letters and constraint[i + 1] in clock_letters:
                loc1 = clock_letters.index(constraint[i]) % 12
                loc2 = clock_letters.index(constraint[i + 1]) % 12
                if (loc2 - loc1) % 12 > 5:
                    return False
            elif constraint[i] in clock_letters:
                completed = False
                loc = clock_letters.index(constraint[i]) % 12
                space = False
                for i in range(1, 6):
                    newloc = (loc + i) % 12
                    if clock_letters[newloc] == 'z' or clock_letters[newloc + 12] == 'z':
                        space = True
                if space == False:
                    return False
            elif constraint[i + 1] in clock_letters:
                completed = False 
                loc = clock_letters.index(constraint[i + 1]) % 12
                space = False
                for i in range(1, 6):
                    newloc = (loc - i) % 12
                    if clock_letters[newloc] == 'z' or clock_letters[newloc + 12] == 'z':
                        space = True
                if space == False:
                    return False
            else:
                completed = False
        return not completed
    
    
    def _build_constraint_repr(self, constraint, useful_letters, clock_letters):
        '''Build string representation of constraint'''
        repr = 'EE'  # pad with two "Edge"s
        for letter in constraint:
            if letter in useful_letters:
                repr += 'O'  # "Owned" letter
            elif letter in clock_letters:
                repr += 'P'  # "Played" letter, i.e. already on the clock
            else:
                repr += '_'  # neither owned nor played
        repr += 'EE'
        return repr


    def play_letter_optimally(self, chosen_letter_stats, clock_letters, clock_spaces_after, clock_spaces_before, clock_density):  # returns chosen_hour
        '''Given a useful letter's stats, return the optimal place to play it'''
        
        chosen_letter = chosen_letter_stats.letter
        viable_cells = set(get_all_empty(clock_letters))
        
        # Restrict viable cells to those which satisfy the constraints the chosen letter belongs to
        for constraint, constraint_repr in chosen_letter_stats.playable_constraint_tups:
            
            constraint_repr_index = constraint.index(chosen_letter) + 2
            viable_cells_restricted = viable_cells.copy()
            
            # Restrict viable cells to those within 5 spaces of a played letter
            for d in [-1, 1]:
                i = constraint_repr_index - d
                if constraint_repr[i] != 'P':
                    continue
                played_hour, _ = get_location(clock_letters, constraint[i - 2])
                viable_cells_restricted &= set([(hour, slot) for hour in crange(played_hour + d, played_hour + d * 6, d, 12) for slot in [0, 1]])
            
            # If there is a further played letter, don't play within a 1-hour radius of it
            for d1, d2 in [(-2, -1), (2, 1)]:
                i, j = (constraint_repr_index - d for d in (d1, d2))
                if not (constraint_repr[i] == 'P' and constraint_repr[j] == 'O'):
                    continue
                played_hour, _ = get_location(clock_letters, constraint[i - 2])
                viable_cells_restricted -= set([(hour, slot) for hour in crange(played_hour - 1, played_hour + 2, 1, 12) for slot in [0, 1]])
            
            # Apply restrictions for this constraint, unless doing so (i.e. to satisfy a smaller constraint) would eliminate all viable cells
            if len(viable_cells_restricted) > 0:
                viable_cells = viable_cells_restricted
            
        # Choose how to sort the viable cells based on whether we will need to place further letters for this constraint later
        lpc, lpc_repr = chosen_letter_stats.playable_constraint_tups[0]
        lpc_repr_index = lpc.index(chosen_letter) + 2
        if any([lpc_repr[lpc_repr_index + d] == 'O' for d in [-1, 1]]):
            metric = clock_spaces_after if lpc_repr[lpc_repr_index + 1] == 'O' else clock_spaces_before
        else:
            metric = {cell: -density for cell, density in clock_density.items()}  # want max sparsity
        
        chosen_hour = max(viable_cells, key=lambda cell: metric[cell])[0]
        return chosen_hour


    def make_backup_play(self, active_constraints, useful_letters, discard_letters, clock_letters):  # returns chosen_hour, chosen_letter
        available_hours = np.where(np.array(clock_letters) == 'Z')
        chosen_hour = self.rng.choice(available_hours[0])
        # if there's no active constraint, then we can play any letter at the densest hour
        if not active_constraints:
            # print("OMGGGGGGG")
            densest_hour = [0]*12
            for i in range(24):
                if clock_letters[i] == 'Z':
                    densest_hour[i%12] += 1
            chosen_letter = self.rng.choice(discard_letters)
            for i in range(12):
                if densest_hour[i] == 1:
                    chosen_hour = i if i in available_hours[0] else i+12
                    break
            return chosen_hour, chosen_letter
        if discard_letters:
            chosen_letter = self.rng.choice(discard_letters)
            # we don't want to play in spots that would block our constraints
            # we want to choose the hour that is outside of the range of our contraints
        else:  # If there's no discard, then we have to play a useful, starting from the shortest constraint
            chosen_letter = None
            for constraint in reversed(active_constraints):
                intersection = set(useful_letters) & set(constraint)
                if intersection:
                    chosen_letter = intersection.pop()
                    break
            if chosen_letter == None:
                chosen_letter = self.rng.choice(useful_letters)
        return chosen_hour, chosen_letter
    
    
def get_letter(clock_letters, hour, slot):  # returns letter (hour in {1, ..., 12}, slot in {0, 1})
    return clock_letters[slot * 12 + hour % 12]

def set_letter(clock_letters, hour, slot, letter):  # returns None (hour in {1, ..., 12}, slot in {0, 1})
    clock_letters[slot * 12 + hour % 12] = letter

def get_location(clock_letters, letter):  # returns hour, slot
    letter_index = clock_letters.index(letter)
    return (letter_index - 1) % 12 + 1, letter_index // 12

def get_all_empty(clock_letters):  # returns list of (hour, slot) tuples
    return [((i - 1) % 12 + 1, i // 12) for i in range(len(clock_letters)) if clock_letters[i] == 'Z']

def is_empty(clock_letters, hour, slot):  # returns whether the given slot of the given hour is unoccupied
    return get_letter(clock_letters, hour, slot) == 'Z'

def crange(start, stop, step, modulo):  # returns circular range iterator
    for i in range(start, stop, step):
        yield i % modulo



class UsefulLetterStats:
    '''Ccomputes stats for the given letter that help us prioritize which one to play'''

    def __init__(self, active_constraints, active_constraint_reprs, letter):

        self.letter = letter

        self.constraint_tups = []
        self.playable_constraint_tups = []
        
        #LPC = Longest Playable Constraint, i.e. longest constraint which is presently suitable to play the letter for
        self.lpc_len = 0
        self.lpc_dist_to_played_letter = 9
        self.lpc_dist_to_edge = 9
        self.lpc_pos = 0

        for constraint, constraint_repr in zip(active_constraints, active_constraint_reprs):
            
            if letter not in constraint:
                continue
            
            self.constraint_tups.append((constraint, constraint_repr))

            # Ensure on the left/right we're either up against an edge or played letter, or one owned letter away from one of those or another owned letter
            constraint_repr_index = constraint.index(letter) + 2
            r, i = constraint_repr, constraint_repr_index
            if not ((r[i-1] in ['P', 'E'] or r[i-2:i] in ['EO', 'PO', 'OO']) and (r[i+1] in ['P', 'E'] or r[i+1:i+3] in ['OE', 'OP', 'OO'])):
                continue

            self.playable_constraint_tups.append((constraint, constraint_repr))

            if len(constraint) <= self.lpc_len:
                continue

            self.lpc_len = len(constraint)
            self.lpc_pos = constraint.index(letter)

            for s in [range(constraint_repr_index + 1, len(constraint_repr)), range(constraint_repr_index - 1, 0, -1)]:
                for j in s:
                    if constraint_repr[j] == '_':
                        pass
                    elif constraint_repr[j] == 'P':
                        self.lpc_dist_to_played_letter = abs(j - constraint_repr_index)
                    elif constraint_repr[j] == 'E':
                        self.lpc_dist_to_edge = abs(j - constraint_repr_index)
                    else:
                        continue
                    break
