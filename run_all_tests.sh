#!/bin/bash

# ----------------------------------------------
# AUTOMATED TEST RUNNER FOR GPSA ASSIGNMENT
# Runs all Taskloop + Explicit Task configurations
# ----------------------------------------------

OUTPUT="results.txt"
EXEC="./gpsa"
X="X2.txt"
Y="Y2.txt"

echo "GPSA AUTOMATED TEST RESULTS" > $OUTPUT
echo "==============================" >> $OUTPUT
echo "" >> $OUTPUT

run_test() {
    MODE=$1
    GRAIN=$2
    LABEL=$3

    echo "Running: $LABEL"
    echo "---- $LABEL ----" >> $OUTPUT

    srun --nodes=1 $EXEC --exec-mode $MODE --grain-size $GRAIN --x $X --y $Y --print-runtime-only >> $OUTPUT

    echo "" >> $OUTPUT
}

# ----------------------------------------------
# TASKLOOP TESTS
# ----------------------------------------------
echo "Running Taskloop tests..."

run_test 2 1   "Taskloop: Grain 1"
run_test 2 4   "Taskloop: Grain 4"
run_test 2 16  "Taskloop: Grain 16"
run_test 2 64  "Taskloop: Grain 64"
run_test 2 256 "Taskloop: Grain 256"
run_test 2 1024 "Taskloop: Grain 1024"
run_test 2 4096 "Taskloop: Grain 4096"

# ----------------------------------------------
# EXPLICIT TASK TESTS
# ----------------------------------------------
echo "Running Explicit Task tests..."

run_test 3 1   "Explicit Tasks: Grain 1"
run_test 3 4   "Explicit Tasks: Grain 4"
run_test 3 16  "Explicit Tasks: Grain 16"
run_test 3 64  "Explicit Tasks: Grain 64"
run_test 3 256 "Explicit Tasks: Grain 256"
run_test 3 1024 "Explicit Tasks: Grain 1024"
run_test 3 4096 "Explicit Tasks: Grain 4096"

echo "All tests completed! Results saved to results.txt"
