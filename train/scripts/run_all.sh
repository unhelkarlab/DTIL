#!/bin/bash

MAX_SESSIONS=15

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
            "LaborDivision2"|"LaborDivision3")
                base="LaborDivision_base"
                ;;
            "Protoss5v5"|"Terran5v5")
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

# Function to wait until there are less than MAX_SESSIONS tmux sessions running
wait_for_free_slot() {
    while true; do
        # Get the current number of running tmux sessions
        running_sessions=$(tmux ls 2>/dev/null | wc -l)

        # If running sessions are less than the max allowed, break the loop
        if [ "$running_sessions" -lt "$MAX_SESSIONS" ]; then
            break
        fi

        # Otherwise, wait for a second before checking again
        sleep 5
    done
}

# Variables
envs=("LaborDivision2" "LaborDivision3" "PO_Movers-v2" "PO_Flood-v2" "Protoss5v5" "Terran5v5")
algs=("maogail" "magail" "mahil" "iiql" "bc")

exp="1015"
seed_max=3

# Skip list for already completed experiments
skip_experiments() {
    local env=$1
    local alg=$2
    local sv=$3
    local seed=$4

    return 1  # Do not skip
}

# Loop through combinations of env, alg, and sv
for env in "${envs[@]}"; do
    for alg in "${algs[@]}"; do
        # Determine base and dim_c based on env and alg
        determine_base $env $alg

        for sv in "${svs[@]}"; do
            for seed in `seq ${seed_max}`; do
                # Check if the experiment should be skipped
                if skip_experiments "$env" "$alg" "$sv" "$seed"; then
                    echo "Skipping already completed experiment: env=${env}, alg=${alg}, sv=${sv}, seed=${seed}"
                    continue
                fi

                # Replace periods in sv with underscores to make valid tmux session names
                sv_clean=$(echo $sv | sed 's/\./_/g')

                # Wait for an available slot (if running sessions >= MAX_SESSIONS)
                wait_for_free_slot

                # Create a unique tmux session name for each experiment
                session_name="${env}_${alg}_sv${sv_clean}_seed${seed}"

                echo "Running in tmux session: ${session_name}"

                # Run the experiment in a new detached tmux session, even inside another tmux session
                tmux new-session -d -s ${session_name} "bash -c 'source ~/venvs/DTIL/bin/activate && python train/run_algs.py alg=${alg} env=${env} base=${base} tag='${exp}Seed${seed}Sv${sv}' supervision=${sv} seed=${seed}'"
            done
        done
    done
done
