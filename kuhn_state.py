import copy

class KuhnState:
    def __init__(self, cards, history="", bets=(1, 1), player=0):
        self.cards = cards          # [p0_card, p1_card]
        self.history = history      # "", "c", "b", "cb", ...
        self.bets = list(bets)
        self.player = player        # 0 or 1

    def is_terminal(self):
        return self.history in ["cc", "bc", "cbc", "bf", "cbf"]

    def payoff(self):
        pot = sum(self.bets)
        if self.history.endswith("f"):
            winner = 1 - self.player
        else:
            winner = 0 if self.cards[0] > self.cards[1] else 1

        return pot - self.bets[winner] if winner == 0 else -(pot - self.bets[1])

    def legal_actions(self):
        if self.history == "":
            return ["c", "b"]
        if self.history == "c":
            return ["c", "b"]
        if self.history in ["b", "cb"]:
            return ["f", "c"]
        return []

    def next_state(self, action):
        new = copy.deepcopy(self)
        new.history += action
        if action in ["b", "c"] and self.history in ["b", "cb"]:
            new.bets[self.player] += 1
        elif action == "b":
            new.bets[self.player] += 1

        new.player = 1 - self.player
        return new

    def infoset(self):
        return self.cards[self.player], self.history
