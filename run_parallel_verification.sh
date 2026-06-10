#!/bin/bash
set -e

cd ~/trust_ai/assignment4

mkdir -p results/parallel_logs

run_one () {
    GPU_ID=$1
    CONFIG=$2

    NAME=$(basename "$CONFIG" .yaml)
    OUT_DIR="results/${NAME}"
    TERMINAL_LOG="results/parallel_logs/${NAME}.terminal.log"

    mkdir -p "$OUT_DIR"

    echo "=========================================="
    echo "[START] GPU ${GPU_ID} | ${CONFIG}"
    echo "[OUT]   ${OUT_DIR}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${GPU_ID} python test.py \
        --config "${CONFIG}" \
        --out_dir "${OUT_DIR}" \
        > "${TERMINAL_LOG}" 2>&1

    echo "[DONE] GPU ${GPU_ID} | ${CONFIG}"
}

# GPU 0: FashionMNIST small/medium eps
(
run_one 0 configs/fashion_mnist_eps_1_255.yaml
run_one 0 configs/fashion_mnist_eps_2_255.yaml
run_one 0 configs/fashion_mnist_eps_4_255.yaml
) &

# GPU 1: FashionMNIST larger eps
(
run_one 1 configs/fashion_mnist_eps_6_255.yaml
run_one 1 configs/fashion_mnist_eps_8_255.yaml
run_one 1 configs/fashion_mnist_eps_12_255.yaml
run_one 1 configs/fashion_mnist_eps_16_255.yaml
) &

# GPU 2: MNIST small eps
(
run_one 2 configs/mnist_fc_eps_0.01.yaml
run_one 2 configs/mnist_fc_eps_0.03.yaml
run_one 2 configs/mnist_fc_eps_0.05.yaml
) &

# GPU 3: MNIST larger eps
(
run_one 3 configs/mnist_fc_eps_0.1.yaml
run_one 3 configs/mnist_fc_eps_0.2.yaml
) &

wait

echo "=========================================="
echo "All verification jobs finished."
echo "Summaries:"
echo "=========================================="

find results -name "results_summary.txt" -print | sort | while read f; do
    echo ""
    echo "########## ${f} ##########"
    cat "$f"
done > results/parallel_summary.txt

cat results/parallel_summary.txt