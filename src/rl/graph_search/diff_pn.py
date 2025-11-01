"""
DiffMLP: Diffusion-based Multi-hop Link Prediction Policy Network.
MLP-based policy with a conditional diffusion denoising process over the action space.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.ops as ops
from src.utils.ops import var_cuda, zeros_var_cuda
from src.rl.graph_search.pn import GraphSearchPolicy


# ---------------------------------------------------------------------------
# Helper: sinusoidal time embedding (fixed, no trainable params)
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Classic transformer sinusoidal positional encoding adapted for diffusion timesteps.
    :param timesteps: (B,) integer tensor of diffusion steps
    :param dim: embedding dimension
    :return: (B, dim) float tensor
    """
    assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / (half - 1)
    )  # (half,)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    return emb


# ---------------------------------------------------------------------------
# Cosine noise schedule
# ---------------------------------------------------------------------------

class CosineSchedule:
    """
    Cosine variance schedule for the forward diffusion process.
    beta_t = 1 - cos(pi*t / (2*T))
    alpha_t = prod_{s=1}^{t} (1 - beta_s)
    """

    def __init__(self, T_max: int, device='cuda'):
        self.T_max = T_max
        self.device = device
        t = torch.arange(1, T_max + 1, dtype=torch.float32, device=device)
        beta = 1.0 - torch.cos(math.pi * t / (2.0 * T_max))
        beta = torch.clamp(beta, min=1e-5, max=0.999)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)         # cumulative product
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)

        self.beta = beta           # (T_max,)
        self.alpha = alpha         # (T_max,)
        self.alpha_bar = alpha_bar # (T_max,)
        self.alpha_bar_prev = alpha_bar_prev
        # posterior variance
        self.sigma2 = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar + 1e-8)

    def get(self, t_int: int):
        """Return scalars for step t (1-indexed)."""
        idx = t_int - 1
        return (
            self.beta[idx],
            self.alpha[idx],
            self.alpha_bar[idx],
            self.sigma2[idx],
        )


# ---------------------------------------------------------------------------
# Conditional Encoder (query + rule + LSTM path)
# ---------------------------------------------------------------------------

class ConditionalEncoder(nn.Module):
    """
    Produces the conditioning vector X_c from prior knowledge.
    Comprises:
      1. Query prior encoding  (source entity + query relation)
      2. Answer prior encoding  (mean pooled answer/entity embeddings)
      3. Relational rule encoding  (MaxPool over weighted transformer-encoded rules)
      4. Path prior encoding  (LSTM over traversed relation-entity pairs)
    """

    def __init__(self, entity_dim, relation_dim, hidden_dim,
                 num_rules=5, rule_len=3, num_transformer_heads=4,
                 num_transformer_layers=1, lstm_layers=1):
        super().__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        action_dim = entity_dim + relation_dim

        # 1. Query prior: [e_s; r_q] -> hidden_dim
        self.query_linear = nn.Linear(entity_dim + relation_dim, hidden_dim)
        self.answer_linear = nn.Linear(entity_dim, hidden_dim)

        # 2. Rule encoding: each rule path is a sequence of relations
        #    We embed each relation and use a small transformer
        self.rule_proj = nn.Linear(relation_dim, hidden_dim)
        # Ensure nhead divides hidden_dim exactly
        _nhead = num_transformer_heads
        while hidden_dim % _nhead != 0 and _nhead > 1:
            _nhead -= 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=_nhead,
            dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True
        )
        self.rule_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 3. Path prior: LSTM over (r_{k-1}, e_{k-1}) pairs
        self.path_lstm = nn.LSTM(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Final fusion
        self.fusion = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, e_s_emb, r_q_emb, answer_emb, rule_embs, rule_confs, path_history):
        """
        :param e_s_emb:    (B, entity_dim)
        :param r_q_emb:    (B, relation_dim)
        :param answer_emb: (B, entity_dim), mean pooled answer embeddings
        :param rule_embs:  (B, N_rules, rule_len, relation_dim) or None
        :param rule_confs: (B, N_rules) confidence scores, or None
        :param path_history: list of (r_emb, e_emb) each (B, dim); may be empty
        :return X_c: (B, hidden_dim)
        """
        B = e_s_emb.size(0)
        device = e_s_emb.device

        # --- query prior ---
        X_q = F.relu(self.query_linear(torch.cat([e_s_emb, r_q_emb], dim=-1)))  # (B, H)
        X_o = F.relu(self.answer_linear(answer_emb))  # (B, H)

        # --- relational rule encoding ---
        if rule_embs is not None and rule_embs.size(1) > 0:
            # rule_embs: (B, N, L, R) -> flatten rules for transformer
            B2, N, L, R = rule_embs.shape
            flat = rule_embs.view(B2 * N, L, R)        # (B*N, L, R)
            flat_proj = self.rule_proj(flat)            # (B*N, L, H)
            encoded = self.rule_transformer(flat_proj)  # (B*N, L, H)
            rule_rep = encoded.mean(dim=1)              # (B*N, H) mean over seq
            rule_rep = rule_rep.view(B2, N, -1)         # (B, N, H)
            # weight by confidence
            if rule_confs is not None:
                weights = rule_confs.unsqueeze(-1)      # (B, N, 1)
                rule_rep = rule_rep * weights
            r_bar = rule_rep.max(dim=1)[0]              # (B, H) MaxPool over rules
        else:
            r_bar = torch.zeros(B, self.hidden_dim, device=device)

        # --- path prior (LSTM) ---
        if len(path_history) > 0:
            # path_history: list of (r_emb, e_emb) each (B, D)
            seq = [torch.cat([r, e], dim=-1) for r, e in path_history]  # each (B, action_dim)
            seq = torch.stack(seq, dim=1)                                 # (B, K, action_dim)
            _, (h_n, _) = self.path_lstm(seq)
            h_path = h_n[-1]   # (B, H) last layer hidden state
        else:
            h_path = torch.zeros(B, self.hidden_dim, device=device)

        # --- fusion ---
        X_c = F.relu(self.fusion(torch.cat([X_q, X_o, r_bar, h_path], dim=-1)))  # (B, H)
        return X_c


# ---------------------------------------------------------------------------
# GAT-based Denoising Layer
# ---------------------------------------------------------------------------

class GATDenoiserLayer(nn.Module):
    """
    Single denoising attention layer.
    Each action embedding is updated based on attention-weighted aggregation
    over all actions in the local action space, conditioned on X_c.
    """

    def __init__(self, action_dim, cond_dim, num_heads=4):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        # Learnable projection for each head
        self.W_a = nn.Linear(action_dim, action_dim, bias=False)
        # Attention vector: scores [W_a(x); X_c]
        self.att_vec = nn.Linear(action_dim + cond_dim, 1, bias=False)
        # Output projection
        self.W_out = nn.Linear(action_dim, action_dim, bias=True)
        self.layer_norm = nn.LayerNorm(action_dim)

    def forward(self, X_t, X_c, action_mask):
        """
        :param X_t:       (B, A, action_dim) noisy action embeddings
        :param X_c:       (B, cond_dim) conditioning vector
        :param action_mask: (B, A) 1 for valid, 0 for padding
        :return: (B, A, action_dim) updated embeddings
        """
        B, A, D = X_t.shape

        # Project actions
        H = self.W_a(X_t)                                # (B, A, D)

        # Expand X_c for each action
        X_c_exp = X_c.unsqueeze(1).expand(B, A, -1)     # (B, A, cond_dim)
        att_input = torch.cat([H, X_c_exp], dim=-1)     # (B, A, D+cond_dim)
        att_scores = self.att_vec(att_input).squeeze(-1) # (B, A)
        att_scores = F.leaky_relu(att_scores, 0.2)

        # Mask padding
        INF = 1e9
        att_scores = att_scores - (1.0 - action_mask) * INF
        alpha = F.softmax(att_scores, dim=-1)            # (B, A)

        # Aggregate
        alpha_exp = alpha.unsqueeze(-1)                  # (B, A, 1)
        agg = (alpha_exp * H).sum(dim=1, keepdim=True)  # (B, 1, D)
        agg = agg.expand(B, A, D)                        # (B, A, D)

        # Residual update
        out = X_t + self.W_out(agg)
        out = self.layer_norm(out)
        return out, alpha


# ---------------------------------------------------------------------------
# Conditional Denoiser (nu_theta)
# ---------------------------------------------------------------------------

class ConditionalDenoiser(nn.Module):
    """
    GAT-based conditional denoiser nu_theta(X_t, X_c, t).
    Predicts the noise added at diffusion step t.
    """

    def __init__(self, action_dim, cond_dim, time_dim=64, num_layers=2):
        super().__init__()
        self.action_dim = action_dim
        self.time_dim = time_dim

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input projection: concatenate action + time_emb -> action_dim
        self.input_proj = nn.Linear(action_dim + cond_dim, action_dim)

        # GAT denoising layers
        self.gat_layers = nn.ModuleList([
            GATDenoiserLayer(action_dim, cond_dim) for _ in range(num_layers)
        ])

        # Output noise prediction
        self.out_proj = nn.Linear(action_dim, action_dim)

    def forward(self, X_t, X_c, t_emb, action_mask):
        """
        :param X_t:        (B, A, action_dim)
        :param X_c:        (B, cond_dim)
        :param t_emb:      (B, time_dim)
        :param action_mask:(B, A)
        :return noise_pred:(B, A, action_dim)
        """
        # Project time embedding and fuse with conditioning
        t_feat = self.time_proj(t_emb)              # (B, cond_dim)
        X_c_t = X_c + t_feat                        # (B, cond_dim) additive fusion

        # Fuse action embs with conditioning for input projection
        B, A, D = X_t.shape
        X_c_t_exp = X_c_t.unsqueeze(1).expand(B, A, -1)   # (B, A, cond_dim)
        H = F.silu(self.input_proj(torch.cat([X_t, X_c_t_exp], dim=-1)))  # (B, A, D)

        # Iterative GAT denoising
        for gat in self.gat_layers:
            H, _ = gat(H, X_c_t, action_mask)

        noise_pred = self.out_proj(H)  # (B, A, D)
        return noise_pred


# ---------------------------------------------------------------------------
# DiffMLP Policy Network
# ---------------------------------------------------------------------------

class DiffMLPPolicy(GraphSearchPolicy):
    """
    DiffMLP policy network that replaces the original MLP-based transit with
    a conditional diffusion denoising process over the local action space.

    Key differences from GraphSearchPolicy.transit():
      - Forward diffusion adds noise to all action embeddings.
      - Reverse diffusion (conditional denoiser) iteratively removes noise.
      - Action selection is based on cosine similarity after denoising.
    """

    def __init__(self, args):
        super().__init__(args)

        # Diffusion hyper-parameters
        self.T_max = args.diff_T_max          # max diffusion steps
        self.gamma_log = args.diff_gamma      # log sensitivity for adaptive T
        self.early_stop_delta = args.diff_delta  # cosine similarity threshold
        self.prior_eta = args.diff_eta        # prior constraint LR
        self.diff_vartheta = args.diff_vartheta  # semantic loss weight
        self.diff_num_layers = args.diff_num_layers  # GAT denoiser layers
        # Trick: temperature scaling for sharper action distribution
        self.score_temperature = getattr(args, 'score_temperature', 1.0)
        # Trick: top-k filtering to remove noisy low-scoring actions
        self.topk_actions = getattr(args, 'topk_actions', 0)
        self.rule_bonus_weight = getattr(args, 'rule_bonus_weight', 0.0)
        self.no_op_penalty = getattr(args, 'no_op_penalty', 0.0)
        self.action_dist_epsilon = getattr(args, 'action_dist_epsilon', 0.0)
        self.max_rules = 5
        self.max_rule_len = args.num_rollout_steps

        # Dimensions
        action_dim = self.action_dim          # entity_dim + relation_dim (or relation_dim)
        # Use action_dim as both condition dim and denoiser dim for consistency
        cond_dim = action_dim                 # condition embedding dim = action_dim

        # Conditional encoder (produces cond_dim = action_dim output)
        self.cond_encoder = ConditionalEncoder(
            entity_dim=args.entity_dim,
            relation_dim=args.relation_dim,
            hidden_dim=cond_dim,
            num_transformer_heads=max(1, cond_dim // 64),
            num_transformer_layers=1,
            lstm_layers=args.history_num_layers,
        )

        # GAT denoiser
        self.denoiser = ConditionalDenoiser(
            action_dim=action_dim,
            cond_dim=cond_dim,
            time_dim=64,
            num_layers=self.diff_num_layers,
        )

        # Action embedding layer norm for stability
        self.action_prior_proj = nn.Linear(action_dim + cond_dim, action_dim)
        self.action_ln = nn.LayerNorm(action_dim)

        # Schedule (will be initialized on first use or during construction)
        self._schedule = None
        self._schedule_device = None
        self.path_history = []
        self._semantic_loss_terms = []
        self._last_semantic_loss = None

    def _get_schedule(self, device):
        if self._schedule is None or self._schedule_device != device:
            self._schedule = CosineSchedule(self.T_max, device=device)
            self._schedule_device = device
        return self._schedule

    def _adaptive_T(self, action_space_size: int) -> int:
        T = min(self.T_max, math.ceil(self.gamma_log * math.log(action_space_size + 1)))
        return max(1, T)

    def _forward_diffuse(self, X0, T, sched, action_space_size):
        """
        Analytically compute X_T from X_0 using closed-form forward diffusion.
        :param X0: (N_actions, action_dim) clean embeddings
        :param T: diffusion steps
        :param sched: CosineSchedule
        :param action_space_size: int, used for normalized noise injection
        :return X_T: (N_actions, action_dim)
        """
        _, _, alpha_bar_T, _ = sched.get(T)
        scale = action_space_size ** 0.5
        noise_scale = math.sqrt(1.0 / (scale + 1e-8))
        noise = torch.randn_like(X0) * noise_scale
        X_T = (alpha_bar_T ** 0.5) * X0 + ((1.0 - alpha_bar_T) ** 0.5) * noise
        return X_T

    def _reverse_step(self, X_t, X_c, t, sched, action_mask, t_emb):
        """
        One reverse diffusion step: predict noise, compute mean, sample.
        :param X_t:       (B, A, D)
        :param X_c:       (B, cond_dim)
        :param t:         int (current step, 1-indexed)
        :param sched:     CosineSchedule
        :param action_mask:(B, A)
        :param t_emb:     (B, time_dim)
        :return X_t_minus1:(B, A, D), mu:(B, A, D)
        """
        beta_t, alpha_t, alpha_bar_t, sigma2_t = sched.get(t)

        # Predict noise
        noise_pred = self.denoiser(X_t, X_c, t_emb, action_mask)  # (B, A, D)

        # Reparameterized mean (Eq. mu_theta)
        mu = (1.0 / (alpha_t ** 0.5)) * (
            X_t - (beta_t / ((1.0 - alpha_bar_t) ** 0.5 + 1e-8)) * noise_pred
        )  # (B, A, D)

        # Sample X_{t-1}
        if t > 1:
            noise = torch.randn_like(X_t)
            X_t1 = mu + (sigma2_t ** 0.5) * noise
        else:
            X_t1 = mu
        return X_t1, mu

    def _apply_prior_constraint(self, X_t, X_prior, action_mask):
        """
        Prior constraint: gradient step to keep noisy action embeddings close to the
        prior context vector X_c.
        X_prior: (B, action_dim)  (cond_dim == action_dim by design)
        X_t:     (B, A, action_dim)
        """
        X_prior_exp = X_prior.unsqueeze(1)            # (B, 1, action_dim)
        diff = X_t - X_prior_exp                      # (B, A, action_dim)
        grad = 2.0 * diff                             # gradient of ||X_t - X_prior||^2
        mask_exp = action_mask.unsqueeze(-1)          # (B, A, 1)
        grad = grad * mask_exp
        X_t_new = X_t - self.prior_eta * grad
        return X_t_new

    def _cosine_sim(self, A, B):
        """Cosine similarity: (B, A, D), (B, A, D) -> (B, A)"""
        A_norm = F.normalize(A, dim=-1)
        B_norm = F.normalize(B, dim=-1)
        return (A_norm * B_norm).sum(dim=-1)

    def initialize_path(self, init_action, kg):
        super().initialize_path(init_action, kg)
        self.path_history = [init_action]

    def update_path(self, action, kg, offset=None):
        if offset is not None and self.path_history:
            offset_l = offset.long()
            self.path_history = [(r[offset_l], e[offset_l]) for r, e in self.path_history]
        super().update_path(action, kg, offset=offset)
        self.path_history.append(action)

    def reset_semantic_loss(self):
        self._semantic_loss_terms = []
        self._last_semantic_loss = None

    def finalize_semantic_loss(self, device):
        if self._semantic_loss_terms:
            self._last_semantic_loss = torch.stack(self._semantic_loss_terms).mean()
        else:
            self._last_semantic_loss = torch.tensor(0.0, device=device, requires_grad=False)
        return self._last_semantic_loss

    def _get_rule_priors(self, q, kg):
        """
        Build rule-path embeddings from AnyBURL rules keyed by query relation.
        Returns padded tensors for the conditional encoder.
        """
        if not getattr(kg, 'rule_paths', None):
            return None, None

        B = q.size(0)
        device = q.device
        rule_ids = torch.zeros(B, self.max_rules, self.max_rule_len, dtype=torch.long, device=device)
        rule_confs = torch.zeros(B, self.max_rules, dtype=torch.float32, device=device)

        for b, rel_id in enumerate(q.detach().cpu().tolist()):
            rules = kg.rule_paths.get(int(rel_id), [])[:self.max_rules]
            for rule_idx, (conf, rel_path) in enumerate(rules):
                clipped_path = rel_path[:self.max_rule_len]
                if not clipped_path:
                    continue
                rule_ids[b, rule_idx, :len(clipped_path)] = torch.tensor(
                    clipped_path, dtype=torch.long, device=device)
                rule_confs[b, rule_idx] = conf

        if rule_confs.sum().item() <= 0:
            return None, None
        rule_embs = kg.get_relation_embeddings(rule_ids.view(-1)).view(
            B, self.max_rules, self.max_rule_len, -1)
        return rule_embs, rule_confs

    def _get_relation_prefixes(self, batch_indices):
        prefixes = []
        for b in batch_indices:
            prefix = []
            for r_step, _ in self.path_history[1:]:
                rel = int(r_step[b])
                if rel in [0, 1, 2]:
                    continue
                prefix.append(rel)
            prefixes.append(tuple(prefix))
        return prefixes

    def _get_rule_action_bias(self, q_b, r_space, action_mask, kg, batch_indices):
        if self.rule_bonus_weight <= 0 or not getattr(kg, 'rule_paths', None):
            return torch.zeros_like(action_mask)

        prefixes = self._get_relation_prefixes(batch_indices)
        bias = torch.zeros_like(action_mask)
        for b, query_r in enumerate(q_b.detach().cpu().tolist()):
            prefix = prefixes[b]
            for conf, rel_path in kg.rule_paths.get(int(query_r), []):
                if len(rel_path) <= len(prefix):
                    continue
                if tuple(rel_path[:len(prefix)]) == prefix:
                    next_rel = rel_path[len(prefix)]
                    rel_mask = (r_space[b] == next_rel).float() * action_mask[b]
                    bias[b] = torch.maximum(bias[b], rel_mask * float(conf))
        return bias

    def transit(self, e, obs, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=False):
        """
        Override GraphSearchPolicy.transit with diffusion-based action selection.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # --- Build conditioning vector X_c ---
        E_s = kg.get_entity_embeddings(e_s)    # (B, entity_dim)
        R_q = kg.get_relation_embeddings(q)    # (B, relation_dim)
        if self.training:
            E_o = kg.get_entity_embeddings(e_t)  # target/answer prior during training
        else:
            E_o = torch.zeros_like(E_s)

        # Extract path history from LSTM path state
        path_history = self._build_path_history(kg)
        rule_embs, rule_confs = self._get_rule_priors(q, kg)

        X_c = self.cond_encoder(
            e_s_emb=E_s,
            r_q_emb=R_q,
            answer_emb=E_o,
            rule_embs=rule_embs,
            rule_confs=rule_confs,
            path_history=path_history,
        )  # (B, cond_dim)

        def policy_diffusion_fun(X_c_b, action_space_b, e_b, q_b, batch_indices_b):
            """Run diffusion denoising for one bucket."""
            (r_space, e_space), action_mask = action_space_b
            B_b, A = r_space.shape
            device = r_space.device

            # Build action embeddings X0: concat relation + entity
            r_emb = kg.get_relation_embeddings(r_space.view(-1)).view(B_b, A, -1)   # (B_b, A, R)
            e_emb = kg.get_entity_embeddings(e_space.view(-1)).view(B_b, A, -1)     # (B_b, A, E)
            if self.relation_only:
                X0_base = r_emb
            else:
                X0_base = torch.cat([r_emb, e_emb], dim=-1)   # (B_b, A, action_dim)

            X_c_actions = X_c_b.unsqueeze(1).expand(B_b, A, -1)
            X0 = self.action_prior_proj(torch.cat([X_c_actions, X0_base], dim=-1))
            X0 = self.action_ln(X0)
            action_mask_f = action_mask.float()           # (B_b, A)

            # Adaptive T
            valid_counts = action_mask_f.sum(dim=-1)      # (B_b,)
            avg_count = max(1, int(valid_counts.mean().item()))
            T_adapt = self._adaptive_T(avg_count)

            sched = self._get_schedule(device)

            # Forward diffusion: X0 -> X_T (batch-wise)
            # Normalize per sample for stability
            X_T_list = []
            for b in range(B_b):
                cnt = max(1, int(valid_counts[b].item()))
                xt = self._forward_diffuse(X0[b], T_adapt, sched, cnt)
                X_T_list.append(xt)
            X_t = torch.stack(X_T_list, dim=0)   # (B_b, A, D)
            X_T = X_t.detach()

            # Reverse diffusion
            t_dim = 64
            hat_t = 1
            X_prior = X_c_b   # (B_b, cond_dim) used as prior constraint reference

            for t in range(T_adapt, 0, -1):
                t_tensor = torch.full((B_b,), t, dtype=torch.long, device=device)
                t_emb = sinusoidal_embedding(t_tensor, t_dim)   # (B_b, t_dim)

                # Prior constraint step
                with torch.no_grad():
                    X_t = self._apply_prior_constraint(X_t, X_prior, action_mask_f)

                # Reverse step
                X_t, mu = self._reverse_step(X_t, X_c_b, t, sched, action_mask_f, t_emb)
                hat_t = t
                # Trick: clamp embeddings to prevent overflow
                X_t = torch.clamp(X_t, -10.0, 10.0)

                # Early stopping check
                cos_sim = self._cosine_sim(mu, X_t)   # (B_b, A)
                cos_sim_masked = cos_sim * action_mask_f
                max_sim = cos_sim_masked.max(dim=-1)[0]  # (B_b,)
                if (max_sim >= self.early_stop_delta).all():
                    break

            if self.training:
                self._semantic_loss_terms.append(
                    self.semantic_consistency_loss(X_T, X_t, action_mask_f))

            # --- Action selection via cosine similarity to X_c ---
            # Eq. (r_k, e_k) ~ Softmax(sim(mu(X_hat_t), X_c))
            mu_norm = F.normalize(X_t, dim=-1)                 # (B_b, A, action_dim)
            X_c_norm = F.normalize(X_c_b, dim=-1)             # (B_b, action_dim)
            scores = (mu_norm * X_c_norm.unsqueeze(1)).sum(-1)  # (B_b, A)
            if self.rule_bonus_weight > 0:
                rule_bias = self._get_rule_action_bias(q_b, r_space, action_mask_f, kg, batch_indices_b)
                scores = scores + self.rule_bonus_weight * rule_bias
            if self.no_op_penalty > 0 and not last_step:
                scores = scores - self.no_op_penalty * (r_space == kg.self_edge).float()

            # Trick: top-k filtering - zero out low-scoring actions before softmax
            if self.topk_actions > 0:
                base_action_mask = action_mask_f
                masked_scores = scores - (1.0 - base_action_mask) * ops.HUGE_INT
                k = min(self.topk_actions, scores.size(-1))
                topk_val = masked_scores.topk(k, dim=-1).values[:, -1:]  # (B_b, 1) threshold
                topk_mask = (scores >= topk_val).float()
                action_mask_f = base_action_mask * topk_mask
                empty_rows = action_mask_f.sum(dim=-1, keepdim=True) <= 0
                action_mask_f = torch.where(empty_rows, base_action_mask, action_mask_f)

            # Mask and temperature-scaled softmax
            scores = scores / max(self.score_temperature, 1e-3)  # temperature scaling
            scores = scores - (1.0 - action_mask_f) * ops.HUGE_INT
            action_dist = F.softmax(scores, dim=-1)             # (B_b, A)
            action_dist = torch.nan_to_num(action_dist, nan=0.0, posinf=0.0, neginf=0.0)
            action_dist = action_dist * action_mask_f
            if self.action_dist_epsilon > 0:
                valid_uniform = action_mask_f / (action_mask_f.sum(dim=-1, keepdim=True) + 1e-8)
                eps = min(max(self.action_dist_epsilon, 0.0), 0.1)
                action_dist = (1.0 - eps) * action_dist + eps * valid_uniform
            action_dist = action_dist / (action_dist.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = ops.entropy(action_dist)
            return action_dist, entropy

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = ops.pad_and_cat(db_r_space, padding_value=kg.dummy_r)[inv_offset]
            e_space = ops.pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
            action_mask = ops.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
            return ((r_space, e_space), action_mask)

        if use_action_space_bucketing:
            db_outcomes = []
            entropy_list = []
            references = []
            db_action_spaces, db_references = self.get_action_space_in_buckets(e, obs, kg)
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                X_c_b = X_c[reference_b, :]
                e_b = e[reference_b]
                q_b = q[reference_b]
                action_dist_b, entropy_b = policy_diffusion_fun(
                    X_c_b, action_space_b, e_b, q_b, reference_b)
                references.extend(reference_b)
                db_outcomes.append((action_space_b, action_dist_b))
                entropy_list.append(entropy_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
            if merge_aspace_batching_outcome:
                db_action_dist = []
                for _, action_dist in db_outcomes:
                    db_action_dist.append(action_dist)
                action_space = pad_and_cat_action_space(db_action_spaces, inv_offset)
                action_dist = ops.pad_and_cat(db_action_dist, padding_value=0)[inv_offset]
                db_outcomes = [(action_space, action_dist)]
                inv_offset = None
        else:
            action_space = self.get_action_space(e, obs, kg)
            action_dist, entropy = policy_diffusion_fun(X_c, action_space, e, q, list(range(len(e))))
            db_outcomes = [(action_space, action_dist)]
            inv_offset = None

        return db_outcomes, inv_offset, entropy

    def _build_path_history(self, kg):
        """
        Extract concrete (relation, entity) actions traversed so far and convert
        them into embeddings for the conditional path-prior LSTM.
        """
        if not self.path_history:
            return []
        history = []
        for r, e in self.path_history:
            history.append((kg.get_relation_embeddings(r), kg.get_entity_embeddings(e)))
        return history

    def semantic_consistency_loss(self, X_T, X_hat_t, action_mask=None):
        """
        L_g = ||X_T - X_hat_t||^2_2
        :param X_T:     (B, A, D) noisy embeddings at step T
        :param X_hat_t: (B, A, D) recovered embeddings at step hat_t
        :return: scalar loss
        """
        loss = (X_T - X_hat_t).pow(2)
        if action_mask is not None:
            mask = action_mask.unsqueeze(-1)
            return (loss * mask).sum() / (mask.sum() * loss.size(-1) + 1e-8)
        return loss.mean()
