#!/bin/bash
set -euox pipefail

# --- Positional arguments ---
INPUT_DIR=$1
OUTPUT_DIR=$2
DATASET_TYPE=$3
PT_SET=$4
DPA_METRIC=$5
MODEL_NAME=$6
DESC_COL=$7
SEL_METRIC=$8
N_TRIALS=$9
N_JOBS=${10}
TEST_FOLD=${11}

PYTHON_PATH="${HOME}/miniforge3/envs/pmd/bin/python"
SCRIPT_PATH="${HOME}/Repos/PMD/src/CLI/optuna_optim.py"

"$PYTHON_PATH" -c "import sklearn, xgboost, optuna, numpy, pandas, polars, scipy; print('Packages OK')"

echo "Using Python at ${PYTHON_PATH}"
echo "Executing script at ${SCRIPT_PATH}"

OMP_NUM_THREADS="${N_JOBS}" MKL_NUM_THREADS="${N_JOBS}" NUMEXPR_NUM_THREADS="${N_JOBS}" "${PYTHON_PATH}" -u "${SCRIPT_PATH}" \
    --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}" --dataset_type "${DATASET_TYPE}" --pt_set "${PT_SET}" \
    --dpa_metric "${DPA_METRIC}" --model_name "${MODEL_NAME}" --desc_col "${DESC_COL}" --sel_metric "${SEL_METRIC}" \
    --n_trials "${N_TRIALS}" --n_jobs "${N_JOBS}" --test_fold "${TEST_FOLD}"
