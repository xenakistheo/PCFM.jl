
What I am wondering about is how I should format sample_pcfm. 
I would like to make it so general that it can be used for all the problems. 
Is it going to take in a constraint function 'H' as an argument?
If so, how should H look like?

Would be beneficial to look at the different BVPs, and the constrains we want to impose on them.

1. Look at BVPs, and try to define their constraints in a general way.
2. Look at whats common between them. 
3. Look at how to define this across the batches so as to leverage the parallelism of the GPU.


Which BVPs do we have?

Paramters for constraints. u_flat, dx, dy, dt, Nx, Ny, Nt. 
All except for u_flat can be passed as 'params' to the constraint function.

### Heat eq. 
1 dimension. Nx = 100, Nt = 100. 
constraint. 




    Other stuff. 
What was up with using a 'Gaussian process'?
Need to train models for each BVP. 


Todo:
test current scripts for heatequation.
Check that constraints are being applied correctly.
Check that training data is being generated correctly for all BVPs. 
Find out how to train models for each BVP.



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