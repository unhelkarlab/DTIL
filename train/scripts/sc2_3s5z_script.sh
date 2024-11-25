#!/bin/bash

determine_base() {
    env=$1
    alg=$2

    # Set base depending on the env and alg
    if [[ "$alg" == "bc" ]]; then
        base=""  # If alg is 'bc', base should be empty
    else
        case "$env" in
            "PO_Flood-v2"|"PO_Movers-v2")
                base="DiscreteWorld_base"
                ;;
            "LaborDivision2-v2"|"LaborDivision3-v2")
                base="LaborDivision_base"
                ;;
            "sc2_2s3z"|"sc2_3s5z")
                base="SC2_base"
                ;;
            *)
                base=""  # Default value if env doesn't match the specified ones
                ;;
        esac
    fi

    # Only set possible supervisions if alg is 'mahil' or 'maogail'
    if [[ "$alg" == "mahil" || "$alg" == "maogail" ]]; then
        svs=("0.0" "0.2")
    else
        svs=("0.0")
    fi
}

# Variables
# envs=("PO_Movers-v2" "LaborDivision2-v2" "PO_Flood-v2")
envs=("sc2_3s5z")
algs=("mahil" "iiql" "magail" "maogail" "bc")

exp="test"
seed_max=3

# Loop through combinations of env, alg, and sv
for env in "${envs[@]}"; do
    for alg in "${algs[@]}"; do
        # Determine base and dim_c based on env and alg
        determine_base $env $alg

        for sv in "${svs[@]}"; do
            for seed in `seq ${seed_max}`; do

                # Create a unique tmux session name for each experiment
                session_name="${env}_${alg}_sv${sv}_seed${seed}"

                echo "Running in tmux session: ${session_name}"

                # Run the experiment in a new detached tmux session, activating the virtual environment first
                tmux new-session -d -s ${session_name} "source ~/venvs/aicoach38/bin/activate && python train_ma_dnn/run_algs.py alg=${alg} env=${env} base=${base} tag='${exp}Seed${seed}Sv${sv}' supervision=${sv} seed=${seed}"
            done
        done
    done
done
