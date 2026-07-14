## Notes 


Heat 2 is perhaps completely infeasible. 
MadNLP GPU performs very bad on RD and NS (while the others perform well)

`ssh orcd-login`
`cd projects/18337/PCFM.jl`


`squeue -u txenakis`

### To run experiments 


-------------------------------------
Heat1: (7 min)
ExaGPU, ExaCPU, JuMP_MadNLP, JuMP_IPopt
`sbatch --job-name=pcfm_heat final_scripts/run_inference.sh heat 6`



-------------------------------------
Heat2: (1 hour)
ExaGPU, ExaCPU
`sbatch --job-name=pcfm_heat2 final_scripts/run_inference.sh heat2 6`


-------------------------------------
BurgersBC: (30 min)
ExaGPU, ExaCPU, JuMP_MadNLP, JuMP_IPopt
`sbatch --job-name=pcfm_burgers_BC final_scripts/run_inference.sh burgers_BC 6`


-------------------------------------
ReactionDiffusion: (10 min)
ExaGPU, ExaCPU, JuMP_MadNLP, JuMP_IPopt
`sbatch --job-name=pcfm_rd final_scripts/run_inference.sh rd 6`


-------------------------------------
NavierStokes: (5 min)
ExaGPU, ExaCPU, JuMP_MadNLP, JuMP_IPopt
`sbatch --job-name=pcfm_ns final_scripts/run_inference.sh ns 6`



-------------------------------------
Burgers (pure) IC: 
ExaGPU, ExaCPU, JuMP_MadNLP, JuMP_IPopt
`sbatch final_scripts/run_constraintscale.sh --constraint IC --k 5 --run 6`

-------------------------------------
Burgers IC + Mass: 
ExaGPU, ExaCPU, JuMP_MadNLP
`sbatch final_scripts/run_constraintscale.sh --constraint IC_Mass --k 5 --run 6`


-------------------------------------
Burgers IC + Mass + Flux(1): 
ExaGPU, ExaCPU
`sbatch final_scripts/run_constraintscale.sh --constraint IC_Mass_Flux --k 1 --run 6`


-------------------------------------
Burgers IC + Mass + Flux(5): 
ExaGPU, ExaCPU
`sbatch final_scripts/run_constraintscale.sh --constraint IC_Mass_Flux --k 5 --run 6`


-------------------------------------
Burgers IC + Mass + Flux(10): 
ExaGPU
`sbatch final_scripts/run_constraintscale.sh --constraint IC_Mass_Flux --k 10 --run 6`






# Compute
In the original submission, the experiments performed by Theo 
were all done on a compute node requested with 
`salloc -p mit_normal_gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=03:00:00`

Alaina's experiments were performed on a compute node requested with
`XXX`


Do this 
`#SBATCH --gres=gpu:l40s:1`
instead of 
`#SBATCH --gres=gpu:1`


CPU benchmarks were run on the mit_normal CPU partition, while GPU benchmarks were run on the mit_normal_gpu partition with one L40S GPU. Both used 4 allocated CPU cores and 64 GB RAM, with CPU threading fixed via OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, and NUMEXPR_NUM_THREADS.




