def ev(constraints: list, hand: str):
    value = dict()
    what2keep = list()
    # evaluating the expected payoff of each constraint
    for constraint in constraints:
        p = 1.0
        for letter in hand:
            idx = constraint.find(letter)
            if idx!=-1:
                uselessletters.discard(letter)
                if (idx>0 and constraint[idx-1] not in hand)\
                and (idx<len(constraint)-1 and constraint[idx+1] not in hand):
                    p *= 0.94
        for idx in range(1,len(constraint)):
            if (constraint[idx-1] not in hand) and (constraint[idx] not in hand):
                p *= 10/23
            else:
                p *= 0.98
        print(f"constraint: {constraint}")
        print(f"p={p:.5f}")
        value[constraint] = 2.0*p-1.0 if len(constraint)==2 else p-1+p*3.0*2**(len(constraint)-3)
        print(f"ev={value[constraint]:.5f}")
        if value[constraint]>0:
            what2keep.append(constraint)
    # removing contradicting constraints - how?

    # returning useless letters
    uselessletters = {letter for letter in hand}
    for constraint in what2keep:
        for letter in constraint:
            uselessletters.discard(letter)
    return what2keep, uselessletters
    