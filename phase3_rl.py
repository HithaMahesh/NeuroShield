"""
phase3_rl.py
Phase 3 — Tabular Q-Learning self-healing agent.
States: 0=Normal, 1=Anomaly (from Phase 1 binary output).
Actions: 0=Allow, 1=Block, 2=Alert.
Cached to disk by app.py via joblib.
"""

import numpy as np
import random

STATES  = {0: "Normal",  1: "Anomaly"}
ACTIONS = {0: "Allow",   1: "Block",   2: "Alert"}

REWARD_MATRIX = {
    0: {0: +5,  1: -10, 2: -1},   # Normal:  Allow=good, Block=bad
    1: {0: -10, 1: +10, 2: +5},   # Anomaly: Block=great, Alert=ok
}


class Phase3QLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1,
                 episodes=1000, random_state=42):
        self.alpha    = alpha
        self.gamma    = gamma
        self.epsilon  = epsilon
        self.episodes = episodes

        self.n_states  = len(STATES)
        self.n_actions = len(ACTIONS)
        self.n_samples = 0

        random.seed(random_state)
        np.random.seed(random_state)

        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

        self.best_reward  = 0.0
        self.reward_matrix = {
            STATES[s]: {ACTIONS[a]: REWARD_MATRIX[s][a] for a in ACTIONS}
            for s in STATES
        }
        self.policy = {}

    # ── Train ─────────────────────────────────────────────────────────────
    def train(self, phase1, loader):
        y_pred = phase1.predict_batch(loader.X)
        self.n_samples = len(y_pred)

        print(f"[Phase3] Q-Learning over {self.episodes} episodes "
              f"on {self.n_samples} samples…")

        for ep in range(self.episodes):
            indices = np.random.choice(
                self.n_samples, size=min(200, self.n_samples), replace=False
            )
            for idx in indices:
                state  = int(y_pred[idx])
                action = (random.randint(0, self.n_actions - 1)
                          if random.random() < self.epsilon
                          else int(np.argmax(self.Q[state])))
                reward     = REWARD_MATRIX[state][action]
                next_state = random.randint(0, self.n_states - 1)
                best_next  = np.max(self.Q[next_state])
                self.Q[state, action] += self.alpha * (
                    reward + self.gamma * best_next - self.Q[state, action]
                )

        self.best_reward = float(np.max(self.Q))
        self.policy = {
            STATES[s]: ACTIONS[int(np.argmax(self.Q[s]))]
            for s in range(self.n_states)
        }
        print(f"[Phase3] Done. Best reward: {self.best_reward:.2f} | "
              f"Policy: {self.policy}")
        return self

    # ── Inference ─────────────────────────────────────────────────────────
    def act(self, state: int, explore: bool = False) -> str:
        """
        explore=True adds a small chance of picking Alert (sub-optimal action)
        so the visualization shows realistic mixed decisions.
        """
        if explore and state == 1:  # only for anomaly state
            roll = random.random()
            if roll < 0.20:          # 20% of anomalies → Alert
                return ACTIONS[2]    # Alert
        return ACTIONS[int(np.argmax(self.Q[state]))]

    # ── Q-table as serializable dict ──────────────────────────────────────
    def q_table_serializable(self):
        return {
            STATES[s]: {
                ACTIONS[a]: round(float(self.Q[s, a]), 2)
                for a in range(self.n_actions)
            }
            for s in range(self.n_states)
        }