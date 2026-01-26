#!/bin/bash

set -euox pipefail

PYTHON_PATH="${HOME}/miniforge3/envs/pmd/bin/python"
SCRIPT_PATH="${HOME}/Repos/PMD/src/CLI/qcg_submit.py"

N_CPUS=96

OMP_NUM_THREADS="${N_CPUS}" MKL_NUM_THREADS="${N_CPUS}" NUMEXPR_NUM_THREADS="${N_CPUS}" VECLIB_MAXIMUM_THREADS="${N_CPUS}" OPENBLAS_NUM_THREADS="${N_CPUS}" nohup "${PYTHON_PATH}" "${SCRIPT_PATH}" > qcg_submit.log 2>&1 &