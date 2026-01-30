import torch
import torch.optim as optim
import random

from models import RegretNet, PolicyNet
from kuhn_state import KuhnState
from encoding import encode_infoset


class DeepCFR:
    def __init__(self):
        self.regret_net = RegretNet()
        self.policy_net = PolicyNet()

        self.regret_buffer = []
        self.policy_buffer = []

        self.r_opt = optim.Adam(self.regret_net.parameters(), lr=1e-3)
        self.p_opt = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def regret_matching(self, x, legal_actions):
        with torch.no_grad():
            regrets = self.regret_net(x)[:len(legal_actions)]
            regrets = torch.clamp(regrets, min=0)
            if regrets.sum() > 0:
                return regrets / regrets.sum()
            return torch.ones(len(legal_actions)) / len(legal_actions)

    def traverse(self, state):
        if state.is_terminal():
            return state.payoff()

        card, history = state.infoset()
        x = encode_infoset(card, history)

        actions = state.legal_actions()
        strategy = self.regret_matching(x, actions)

        utils = torch.zeros(len(actions))
        for i, a in enumerate(actions):
            utils[i] = self.traverse(state.next_state(a))

        node_util = (strategy * utils).sum()
        regrets = utils - node_util

        self.regret_buffer.append((x, regrets.detach()))
        self.policy_buffer.append((x, strategy.detach()))

        return node_util

    def train_net(self, net, buffer, opt):
        for x, y in buffer:
            loss = ((net(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def train(self, iterations=2000):
        for _ in range(iterations):
            cards = random.sample([0, 1, 2], 2)
            self.traverse(KuhnState(cards))

            self.train_net(self.regret_net, self.regret_buffer, self.r_opt)
            self.train_net(self.policy_net, self.policy_buffer, self.p_opt)

            self.regret_buffer.clear()
            self.policy_buffer.clear()
