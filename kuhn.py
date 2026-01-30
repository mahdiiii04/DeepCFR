import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces


class KuhnPokerEnv(AECEnv):
    metadata = {"name": "kuhn_poker_v0"}

    def __init__(self):
        super().__init__()

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {
            agent: spaces.Discrete(2)  # 0 = check/fold, 1 = bet/call
            for agent in self.agents
        }

        # Observation:
        # [own_card (0–2), has_opponent_bet (0/1)]
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=2, shape=(2,), dtype=np.int32)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        # Deal cards
        deck = [0, 1, 2]
        np.random.shuffle(deck)
        self.cards = {
            "player_0": deck[0],
            "player_1": deck[1],
        }

        self.bets = {a: 1 for a in self.agents}  # antes
        self.history = []
        self.bet_made = False

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        opponent = self.agents[1] if agent == self.agents[0] else self.agents[0]
        return np.array([
            self.cards[agent],
            int(self.bet_made)
        ], dtype=np.int32)

    def step(self, action):
        agent = self.agent_selection
        opponent = self.agents[1] if agent == self.agents[0] else self.agents[0]

        self.history.append((agent, action))

        if not self.bet_made:
            if action == 1:  # bet
                self.bet_made = True
                self.bets[agent] += 1
            else:  # check
                if len(self.history) == 2:
                    self._resolve_showdown()
                    return
        else:
            if action == 0:  # fold
                self.rewards[agent] = -self.bets[agent]
                self.rewards[opponent] = self.bets[agent]
                self._terminate()
                return
            else:  # call
                self.bets[agent] += 1
                self._resolve_showdown()
                return

        self.agent_selection = self._agent_selector.next()

    def _resolve_showdown(self):
        pot = sum(self.bets.values())
        p0, p1 = self.agents

        if self.cards[p0] > self.cards[p1]:
            self.rewards[p0] = pot - self.bets[p0]
            self.rewards[p1] = -self.bets[p1]
        else:
            self.rewards[p1] = pot - self.bets[p1]
            self.rewards[p0] = -self.bets[p0]

        self._terminate()

    def _terminate(self):
        for a in self.agents:
            self.terminations[a] = True
        self.agents = []
