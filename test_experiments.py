# Regression tests for experimental validity, not navigation performance.

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from config import DATA_DIR
from dataset import OfflineDataset
from make_report import load_runs, mean_std
from run_experiments import experiment_grid
from train import (
    build_parser,
    evaluate,
    isolated_rng,
    make_agent,
    policy_q,
    set_seed,
    train,
)


class ExperimentTests(unittest.TestCase):
    # Check fair evaluation, delayed updates and independently saved artifacts.

    @classmethod
    def setUpClass(cls):
        # Keep CPU tests small and independent of the number of host cores.
        torch.set_num_threads(1)
        cls.dataset = OfflineDataset(DATA_DIR / "small.npz")

    def test_evaluation_preserves_rng_and_is_repeatable(self):
        # Evaluation must not change subsequent training batches or VAE draws.
        set_seed(3)
        agent = make_agent("bcq")
        before = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
        )
        first = evaluate(agent, episodes=2)
        second = evaluate(agent, episodes=2)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_bcq_phi_zero_and_checkpoint_phi(self):
        # Phi=0 forbids perturbation; checkpoint loading must restore phi.
        agent = make_agent("bcq", phi=0)
        states, actions, *_ = self.dataset.sample(8)
        torch.testing.assert_close(agent.actor(states, actions), actions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            agent.save(path)
            restored = make_agent("bcq", phi=1)
            restored.load(path)
            self.assertEqual(restored.actor.phi, 0)
            self.assertEqual(restored.actor_target.phi, 0)

    def test_td3_delayed_updates(self):
        # TD3's actor and targets update every second critic step.
        agent = make_agent("td3")
        actor = copy.deepcopy(agent.actor.state_dict())
        target = copy.deepcopy(agent.critic_target.state_dict())
        first = agent.train(self.dataset.sample(8))
        self.assertNotIn("actor_loss", first)
        for name, value in agent.actor.state_dict().items():
            torch.testing.assert_close(value, actor[name], rtol=0, atol=0)
        for name, value in agent.critic_target.state_dict().items():
            torch.testing.assert_close(value, target[name], rtol=0, atol=0)
        second = agent.train(self.dataset.sample(8))
        self.assertIn("actor_loss", second)
        self.assertTrue(
            any(
                not torch.equal(value, actor[name])
                for name, value in agent.actor.state_dict().items()
            )
        )

    def test_td3_terminal_targets(self):
        # True terminals remove bootstrap; nonterminal/timeouts retain it.
        agent = make_agent("td3")
        # Constant online Q=0, target Q=2 makes the expected loss analytical.
        with torch.no_grad():
            for parameter in agent.critic.parameters():
                parameter.zero_()
            for parameter in agent.critic_target.parameters():
                parameter.zero_()
            agent.critic_target.q1_output.bias.fill_(2)
            agent.critic_target.q2_output.bias.fill_(2)
        states = torch.zeros(2, 4)
        actions = torch.zeros(2, 2)
        rewards = torch.tensor([[1.0], [1.0]])
        dones = torch.tensor([[1.0], [0.0]])
        losses = agent.train((states, actions, rewards, states, dones))
        expected = 1.0**2 + (1 + agent.gamma * 2) ** 2
        self.assertAlmostEqual(losses["critic_loss"], expected, places=5)

    def test_policy_q_uses_deployed_policy(self):
        # Q diagnostics must score the same BCQ candidate selection as rollout.
        agent = make_agent("bcq")
        states = self.dataset.states[:4]
        with isolated_rng(5), torch.no_grad():
            expected = (
                agent.critic.q1(states, agent.policy_actions(states)).mean().item()
            )
        with isolated_rng(5):
            observed = policy_q(agent, states)
        self.assertAlmostEqual(observed, expected, places=6)
        self.assertIsNone(policy_q(make_agent("bc"), states))

    def test_grid_and_seed_statistics(self):
        # Ablation shares three main runs; uncertainty is across seeds.
        self.assertEqual(len(experiment_grid("main", [0, 1, 2])), 36)
        self.assertEqual(len(experiment_grid("ablation", [0, 1, 2])), 18)
        self.assertEqual(len(experiment_grid("all", [0, 1, 2])), 51)
        mean, std = mean_std([1, 2, 3])
        self.assertEqual((mean, std), (2, 1))
        self.assertTrue(np.isnan(mean_std([1])[1]))

    def test_run_artifacts_and_report_protocol_guard(self):
        # Runs remain distinct and reports reject mixed training budgets.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_parser().parse_args(
                [
                    "--algo",
                    "bc",
                    "--dataset",
                    str(DATA_DIR / "small.npz"),
                    "--steps",
                    "2",
                    "--eval-interval",
                    "1",
                    "--eval-episodes",
                    "1",
                    "--test-episodes",
                    "3",
                    "--probe-size",
                    "4",
                    "--results-dir",
                    str(root / "runs"),
                    "--checkpoint-dir",
                    str(root / "checkpoints"),
                ]
            )
            first_path, second_path = train(args), train(args)
            self.assertNotEqual(first_path, second_path)
            first = json.loads(first_path.read_text())
            second = json.loads(second_path.read_text())
            self.assertEqual(first["history"], second["history"])
            self.assertNotEqual(first["checkpoint"], second["checkpoint"])
            self.assertTrue(Path(first["checkpoint"]).exists())
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["probe_indices"], second["probe_indices"])
            best = max(
                first["history"], key=lambda row: (row["success_rate"], row["average_reward"])
            )
            self.assertEqual(first["selected_step"], best["step"])
            self.assertEqual(first["final"]["episodes"], 3)
            self.assertEqual(first["generalization"]["episodes"], 3)
            evaluation = first["evaluation"]
            self.assertEqual(
                len({evaluation["seed"], evaluation["test_seed"], evaluation["generalization_seed"]}),
                3,
            )
            self.assertEqual(first["final"], second["final"])
            second["steps"] = 3
            second_path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                load_runs(root / "runs")


if __name__ == "__main__":
    unittest.main()
