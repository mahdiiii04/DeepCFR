import torch
from kuhn import KuhnPokerEnv
from encoding import encode_infoset

def play_episode(policy_net):
    env = KuhnPokerEnv()
    env.reset()

    history = ""

    for agent in env.agent_iter():
        obs = env.observe(agent)
        card, bet_flag = obs

        x = encode_infoset(card, history)
        probs = policy_net(x).detach().numpy()
        action = int(probs.argmax())

        env.step(action)
        history += "b" if action == 1 else "c"

    return env.rewards
