#!/bin/bash
# Run all constraint violation scripts and save combined output to a single txt file.
# Usage: bash final_scripts/run_constraint_violations.sh
# Output: logs/constraint_violations.txt

set -euo pipefail

OUTPUT="logs/constraint_violations.txt"
mkdir -p logs

SCRIPTS=(
    examples/constraintViolations/burgersBC.jl
    examples/constraintViolations/burgersIC.jl
    examples/constraintViolations/burgersICMass.jl
    examples/constraintViolations/burgersICMassFlux1.jl
    examples/constraintViolations/burgersICMassFlux5.jl
    examples/constraintViolations/burgersICMassFlux10.jl
    examples/constraintViolations/heat1.jl
    examples/constraintViolations/heat2.jl
    examples/constraintViolations/ns.jl
    examples/constraintViolations/rd.jl
)

> "$OUTPUT"

for script in "${SCRIPTS[@]}"; do
    name=$(basename "$script")
    echo "============================== $name ==============================" | tee -a "$OUTPUT"
    julia --project=. "$script" 2>&1 | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
done

echo "Done. Results saved to $OUTPUT"
