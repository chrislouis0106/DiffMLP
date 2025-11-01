#!/bin/bash
# DiffMLP experiment script (mirrors experiment.sh but adds diff_* hyperparameters)
# Usage:
#   Train:     bash experiment-diff.sh configs/fb15k-237-diffmlp.sh --train  0
#   Inference: bash experiment-diff.sh configs/fb15k-237-diffmlp.sh --inference 0

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi
relation_only_flag=''
if [[ $relation_only = *"True"* ]]; then
    relation_only_flag="--relation_only"
fi
use_action_space_bucketing_flag=''
if [[ $use_action_space_bucketing = *"True"* ]]; then
    use_action_space_bucketing_flag='--use_action_space_bucketing'
fi

# Tricks flags (optional)
reward_shaping_flag=''
if [[ $reward_shaping = *"True"* ]]; then
    reward_shaping_flag='--reward_shaping'
fi
use_path_reward_flag=''
if [[ $use_path_reward = *"True"* ]]; then
    use_path_reward_flag='--use_path_reward'
fi

# Auto-enable beam search path saving for inference to compute MPS
save_paths_flag=''
if [[ $exp = *"--inference"* ]]; then
    save_paths_flag='--save_beam_search_paths'
fi

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --model $model \
    --bandwidth $bandwidth \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --history_dim $history_dim \
    --history_num_layers $history_num_layers \
    --num_rollouts $num_rollouts \
    --num_rollout_steps $num_rollout_steps \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --num_peek_epochs $num_peek_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --margin $margin \
    --learning_rate $learning_rate \
    --baseline $baseline \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --ff_dropout_rate $ff_dropout_rate \
    --action_dropout_rate $action_dropout_rate \
    --action_dropout_anneal_interval $action_dropout_anneal_interval \
    $relation_only_flag \
    --beta $beta \
    --beam_size $beam_size \
    --num_paths_per_entity $num_paths_per_entity \
    $group_examples_by_query_flag \
    $use_action_space_bucketing_flag \
    --checkpoint_path $checkpoint_path \
    --gpu $gpu \
    --diff_T_max $diff_T_max \
    --diff_gamma $diff_gamma \
    --diff_delta $diff_delta \
    --diff_eta $diff_eta \
    --diff_vartheta $diff_vartheta \
    --diff_num_layers $diff_num_layers \
    ${score_temperature:+--score_temperature $score_temperature} \
    ${topk_actions:+--topk_actions $topk_actions} \
    ${path_reward_weight:+--path_reward_weight $path_reward_weight} \
    ${rule_bonus_weight:+--rule_bonus_weight $rule_bonus_weight} \
    ${no_op_penalty:+--no_op_penalty $no_op_penalty} \
    ${action_dist_epsilon:+--action_dist_epsilon $action_dist_epsilon} \
    ${reward_clip:+--reward_clip $reward_clip} \
    $reward_shaping_flag \
    $use_path_reward_flag \
    $save_paths_flag \
    $ARGS"

echo "Executing $cmd"

$cmd
