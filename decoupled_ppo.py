# ppo_train.py
import torch
import gymnasium as gym
from torch import nn, optim
from torch.distributions import Normal

import matplotlib.pyplot as plt
import re

TOTAL_RUNS = 20  # 训练次数
BASE_SAVE_PATH = "vcppo_pendulum_model"
LOG_FILE = "vcppo_training_log.txt"


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, obs):
        mean, std = self(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    returns, gae, next_value = [], 0, 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
        next_value = values[step]
    return returns


def ppo_train(run_id: int, log_file):
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim)

    pi_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    vf_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    epochs = 100
    steps_per_epoch = 4000
    gamma = 0.99
    lam_actor = 0.95   # used for policy update
    lam_critic = 1.0   # used for value target

    clip_ratio = 0.2
    train_iters = 80
    target_kl = 0.01

    for epoch in range(epochs):
        obs_buf, act_buf, logp_buf = [], [], []
        val_buf, rew_buf, mask_buf = [], [], []

        obs, _ = env.reset()
        for step in range(steps_per_epoch):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, logp = actor.get_action(obs_tensor)
                value = critic(obs_tensor)

            clipped_action = action.clamp(
                torch.tensor(env.action_space.low),
                torch.tensor(env.action_space.high)
            )
            next_obs, reward, terminated, truncated, _ = env.step(clipped_action.numpy())
            done = terminated or truncated

            obs_buf.append(obs_tensor)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value)
            rew_buf.append(reward)
            mask_buf.append(0.0 if done else 1.0)

            obs = next_obs
            if done:
                obs, _ = env.reset()

        with torch.no_grad():
            last_value = critic(torch.tensor(obs, dtype=torch.float32))

        # ✅ Compute two sets of returns:
        adv_actor = compute_gae(rew_buf, val_buf, mask_buf, gamma, lam_actor)
        ret_critic = compute_gae(rew_buf, val_buf, mask_buf, gamma, lam_critic)
        adv_actor = torch.tensor(adv_actor, dtype=torch.float32)
        ret_critic = torch.tensor(ret_critic, dtype=torch.float32)

        # Normalize advantage for policy
        adv_actor = (adv_actor - adv_actor.mean()) / (adv_actor.std() + 1e-8)

        # Stack buffers
        obs_buf = torch.stack(obs_buf)
        act_buf = torch.stack(act_buf)
        logp_buf = torch.stack(logp_buf)

        for _ in range(train_iters):
            mean, std = actor(obs_buf)
            dist = Normal(mean, std)
            new_logp = dist.log_prob(act_buf).sum(axis=-1)

            ratio = torch.exp(new_logp - logp_buf)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            loss_pi = -(torch.min(ratio * adv_actor, clipped_ratio * adv_actor)).mean()

            pi_optimizer.zero_grad()
            loss_pi.backward()
            pi_optimizer.step()

            value_pred = critic(obs_buf)
            loss_v = ((value_pred - ret_critic) ** 2).mean()

            vf_optimizer.zero_grad()
            loss_v.backward()
            vf_optimizer.step()

            kl = (logp_buf - new_logp).mean().item()
            if kl > 1.5 * target_kl:
                print(f"Early stopping at train_iter due to KL: {kl:.4f}")
                break

        avg_reward = sum(rew_buf) / len(rew_buf)
        log_msg = f"[Run {run_id:03d}] Epoch {epoch + 1:03d}, Avg Reward: {avg_reward:.2f}"
        print(log_msg)
        log_file.write(log_msg + "\n")
        log_file.flush()

    # Save model with run ID
    save_path = f"{BASE_SAVE_PATH}_{run_id:03d}.pth"
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
    }, save_path)
    print(f"✅ Saved model to: {save_path}")
    env.close()


if __name__ == "__main__":
    with open(LOG_FILE, "w") as log_file:
        for run_id in range(1, TOTAL_RUNS + 1):
            ppo_train(run_id, log_file)

    # 存储格式：{run_id: [avg_reward1, avg_reward2, ..., avg_rewardN]}
    runs = {}

    with open(LOG_FILE, "r") as f:
        for line in f:
            match = re.match(r"\[Run (\d+)] Epoch \d+, Avg Reward: ([\d\.\-]+)", line)
            if match:
                run_id = int(match.group(1))
                reward = float(match.group(2))
                if run_id not in runs:
                    runs[run_id] = []
                runs[run_id].append(reward)

    # 绘图
    plt.figure(figsize=(10, 6))

    # 每个 run 的曲线（透明一点）
    for run_id, rewards in runs.items():
        plt.plot(rewards, alpha=0.3, label=f"Run {run_id:03d}")

    # 计算平均曲线（按 epoch 平均）
    max_epoch = max(len(r) for r in runs.values())
    reward_matrix = []

    for epoch in range(max_epoch):
        epoch_rewards = [runs[run][epoch] for run in runs if epoch < len(runs[run])]
        reward_matrix.append(sum(epoch_rewards) / len(epoch_rewards))

    plt.plot(reward_matrix, color="black", linewidth=2.5, label="Mean across runs")

    plt.title("Decoupled PPO Training: Avg Reward over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.ylim(-8, 0)
    plt.grid(True)
    plt.legend(loc="upper left", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig("decoupled_ppo_training_plot.png", dpi=200)
    plt.show()
