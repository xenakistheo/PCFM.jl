## Notes 


`ssh orcd-login`
`cd projects/18337/PCFM.jl`


`squeue -u txenakis`

### To run experiments 

Scripts should take in samples path. 

-------------------------------------
Heat1: (7 min)
ExaGPU, ExaCPU, JuMP_MadNLP, JuMP_IPopt
`sbatch --job-name=pcfm_heat final_scripts/run_inference.sh heat 6`
Checked: ok
Ran:


-------------------------------------
Heat2: (1 hour)
ExaGPU, ExaCPU
`sbatch --job-name=pcfm_heat2 final_scripts/run_inference.sh heat2 6`
Checked: ok 
Ran:


-------------------------------------
BurgersBC: (30 min)
ExaGPU, ExaCPU, JuMP_MadNLP, JuMP_IPopt
`sbatch --job-name=pcfm_burgers_BC final_scripts/run_inference.sh burgers_BC 6`
Checked: ok
Ran:

-------------------------------------
ReactionDiffusion: (10 min)
ExaGPU, ExaCPU, JuMP_MadNLP, JuMP_IPopt
`sbatch --job-name=pcfm_rd final_scripts/run_inference.sh rd 6`
Checked: ok
Ran:

-------------------------------------
NavierStokes: (5 min)
ExaGPU, ExaCPU, JuMP_MadNLP, JuMP_IPopt
`sbatch --job-name=pcfm_ns final_scripts/run_inference.sh ns 6`
Checked: ok
Ran:



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




