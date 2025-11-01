#!/usr/bin/env bash
# DiffMLP configuration for FB15K-237

data_dir="data/FB15K237"
model="diffmlp"
group_examples_by_query="False"
use_action_space_bucketing="True"

bandwidth=400
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
bucket_interval=10
num_epochs=100
num_wait_epochs=20
num_peek_epochs=2
batch_size=8
train_batch_size=8
dev_batch_size=2
learning_rate=0.0001
baseline="avg_reward"
grad_norm=5
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.1
action_dropout_anneal_interval=1000
beta=0.02
relation_only="False"
beam_size=128
checkpoint_path="None"

num_paths_per_entity=-1
margin=-1

# DiffMLP specific parameters (optimized for better MPS)
diff_T_max=400
diff_gamma=2.0
diff_delta=0.90              # ↑ from 0.85 for better denoising
diff_eta=0.01                # ↑ from 0.005 for stronger prior constraint
diff_vartheta=0.2            # ↑ from 0.1 for stronger semantic consistency
diff_num_layers=4            # ↓ from 6 for efficiency

# Tricks for improving accuracy (optional, comment out to disable)
score_temperature=0.9        # < 1.0 for sharper action distribution
topk_actions=80              # filter low-scoring actions
rule_bonus_weight=0.15       # encourage actions that continue high-confidence rules
no_op_penalty=0.05           # discourage early NO_OP paths
action_dist_epsilon=1e-6     # smooth valid action probabilities for stable sampling
reward_clip=2.0              # clip R_e + R_p before baseline normalization
reward_shaping="True"        # add intermediate step rewards
use_path_reward="True"       # enable strict rule-path reward R_p
path_reward_weight=0.5       # keep rule reward helpful but not dominant
