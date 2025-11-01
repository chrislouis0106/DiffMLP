"""
DiffMLP Policy Gradient training framework.
Extends PolicyGradient with:
1. Semantic consistency loss L_g
2. Path reward R_p for improving MPS
"""

import torch
import torch.nn.functional as F
import numpy as np

from src.rl.graph_search.pg import PolicyGradient
from src.rules.rule import HornRule
import src.utils.ops as ops
from src.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda, var_to_numpy


class DiffPolicyGradient(PolicyGradient):
    """
    Policy gradient trainer for DiffMLP.
    Implements: L = L_p + ϑ·L_g
    where L_p uses R_K = R_e + λ·R_p
    """

    def __init__(self, args, kg, pn):
        super().__init__(args, kg, pn)
        self.diff_vartheta = args.diff_vartheta   # trade-off weight for L_g
        self.use_path_reward = args.use_path_reward  # whether to use R_p
        self.path_reward_weight = args.path_reward_weight  # λ for R_p
        self.reward_shaping = getattr(args, 'reward_shaping', False)  # intermediate rewards
        self.reward_clip = getattr(args, 'reward_clip', 2.0)
        
        print(f'DiffPolicyGradient initialized:')
        print(f'  - use_path_reward: {self.use_path_reward}')
        print(f'  - path_reward_weight: {self.path_reward_weight}')
        print(f'  - diff_vartheta (L_g weight): {self.diff_vartheta}')
        print(f'  - use_conf: {self.use_conf}')
        print(f'  - support_times: {self.support_times}')

    def extract_paths_from_trace(self, path_trace):
        """
        Extract paths from path_trace for path reward calculation.
        :param path_trace: list of (r_batch, e_batch) tuples
        :return: list of paths, each path is a list of (r, e) tuples
        """
        if not path_trace:
            return []
        
        batch_size = len(path_trace[0][0])
        paths = [[] for _ in range(batch_size)]
        
        for step_idx, (r_batch, e_batch) in enumerate(path_trace):
            for batch_idx in range(batch_size):
                r_val = int(r_batch[batch_idx])
                e_val = int(e_batch[batch_idx])
                paths[batch_idx].append((r_val, e_val))
        
        return paths

    def reward_fun(self, e1, r, e2, pred_e2):
        """
        Entity reward from the method description:
        R_e = 1 for an exact hit; otherwise use an embedding-model score.
        The fallback score uses the trainable KG embeddings with a DistMult-style scorer.
        """
        exact_reward = (pred_e2 == e2).float()
        with torch.no_grad():
            e1_emb = self.kg.get_entity_embeddings(e1)
            r_emb = self.kg.get_relation_embeddings(r)
            pred_emb = self.kg.get_entity_embeddings(pred_e2)
            emb_score = torch.sigmoid(torch.sum(e1_emb * r_emb * pred_emb, dim=-1))
            emb_score = emb_score * (pred_e2 != self.kg.dummy_e).float()
        return torch.where(exact_reward > 0, exact_reward, emb_score)

    def compute_path_reward(self, path_trace, e1, r, e2):
        """
        Strict path reward from the method description:
        R_p = 1 if the sampled relation path tau is in P(r_q), otherwise 0.
        """
        if not self.use_path_reward:
            return var_cuda(torch.zeros(e1.size(0)), requires_grad=False)

        rule_paths = getattr(self.kg, 'rule_paths', {})
        rewards = []
        relation_steps = path_trace[1:]
        for idx in range(e1.size(0)):
            query_r = int(r[idx])
            sampled_path = []
            for r_step, _ in relation_steps:
                rel = int(r_step[idx])
                if rel in [self.kg.dummy_r, self.kg.dummy_start_r, self.kg.self_edge]:
                    continue
                sampled_path.append(rel)
            sampled_path = tuple(sampled_path)
            valid_paths = {tuple(rel_path) for _, rel_path in rule_paths.get(query_r, [])}
            rewards.append(1.0 if sampled_path in valid_paths else 0.0)
        return var_cuda(torch.FloatTensor(rewards), requires_grad=False)

    def loss(self, mini_batch):
        """
        Compute combined loss: L = L_p + ϑ·L_g
        where L_p uses R_K = R_e + λ·R_p
        """
        pn = self.mdl
        
        def stablize_reward(r):
            """Reward stabilization using baseline"""
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'max_min_scalar':
                stabled_r_2D = (r_2D - torch.min(r_2D, dim=1, keepdim=True)[0]) / \
                              (torch.max(r_2D, dim=1, keepdim=True)[0] - torch.min(r_2D, dim=1, keepdim=True)[0] + 1e-8)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / \
                              (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError(f'Unrecognized baseline function: {self.baseline}')
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r
        
        # Format batch
        e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        
        # Rollout
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)
        
        # Extract outputs
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']
        path_trace = output["path_trace"]
        
        # Compute R_e (entity reward)
        baseline_reward = self.reward_fun(e1, r, e2, pred_e2)
        
        # Trick: reward shaping - add small intermediate rewards for valid steps
        intermediate_reward = 0.0
        if hasattr(self, 'reward_shaping') and self.reward_shaping:
            # Small bonus for each valid action (not dummy)
            for r_batch, e_batch in path_trace[1:]:  # skip initial dummy
                valid_mask = (e_batch != self.kg.dummy_e).float()
                intermediate_reward += 0.01 * valid_mask.mean()
        
        # Compute R_p (path reward) if enabled
        if self.use_path_reward:
            path_reward = self.compute_path_reward(path_trace, e1, r, e2)
            # Combined reward: R_K = R_e + λ·R_p + intermediate
            final_reward = baseline_reward + self.path_reward_weight * path_reward + intermediate_reward
        else:
            final_reward = baseline_reward + intermediate_reward
            path_reward = var_cuda(torch.zeros(e1.size(0)), requires_grad=False)
        if self.reward_clip > 0:
            final_reward = torch.clamp(final_reward, 0.0, self.reward_clip)
        
        # Stabilize reward
        final_reward = stablize_reward(final_reward)
        
        # Compute cumulative discounted rewards
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = final_reward
        for i in range(self.num_rollout_steps - 2, -1, -1):
            R = self.gamma * R
            cum_discounted_rewards[i] = R
        
        # Policy gradient loss L_p
        pg_loss = torch.zeros(1, device=e1.device)
        for i in range(self.num_rollout_steps):
            log_prob = log_action_probs[i]
            pg_loss += -torch.sum(cum_discounted_rewards[i] * log_prob)
        
        # Entropy regularization
        entropy = torch.cat(action_entropy).mean()
        pg_loss = (pg_loss - self.beta * entropy) / e1.size(0)
        
        # Semantic consistency loss L_g
        Lg = getattr(pn, '_last_semantic_loss', None)
        if Lg is not None and self.diff_vartheta > 0:
            total_loss = pg_loss + self.diff_vartheta * Lg
        else:
            total_loss = pg_loss
            if Lg is None:
                Lg = torch.tensor(0.0, device=e1.device)
        
        # Return loss dict with detailed metrics
        return {
            'model_loss': total_loss,
            'print_loss': float(total_loss),
            'reward': float(final_reward.mean()),
            'entity_reward': float(baseline_reward.mean()),
            'path_reward': float(path_reward.mean()) if self.use_path_reward else 0.0,
            'semantic_loss': float(Lg),
            'entropy': float(entropy),
        }

    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Override rollout to accumulate semantic consistency loss across hops.
        """
        assert num_steps > 0
        kg, pn = self.kg, self.mdl

        log_action_probs = []
        action_entropy = []
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []
        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg)

        # Reset semantic loss accumulation
        pn.reset_semantic_loss()

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t == (num_steps - 1), last_r, seen_nodes]
            db_outcomes, inv_offset, policy_entropy = pn.transit(
                e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            if visualize_action_probs:
                top_k_action = sample_outcome.get('top_actions', None)
                top_k_action_prob = sample_outcome.get('top_action_probs', None)
                path_components.append((e, top_k_action, top_k_action_prob))

        pn.finalize_semantic_loss(e_s.device)

        pred_e2 = path_trace[-1][1]
        self.record_path_trace(path_trace)

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components,
        }
