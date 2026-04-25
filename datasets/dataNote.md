
Let $\Omega$ be the spatial domain, and $[0,T]$ be the time interval. 
Further, let $Nx$ be the number of spatial grid points - in the $x$ direction, and $Nt$ be the number of time steps. In the case that the PDE is in 2 spatial dimensions, we will also have $Ny$ as the number of grid points in the $y$ direction.
Let $Nb$ be the batch size for training.

We are interested in generating datasets for the following PDEs. 


### Heat equation
$\Omega = [0,2\pi], T=1$, $Nx = 100, Nt = 100$.
$Nb = 32$.

### Navier-Stokes 
$\Omega = [0,2\pi]^2, T=49$, $Nx = Ny = 64, Nt = 50$.
$Nb = 1000$. 

### Reaction-Diffusion
$\Omega = [0,1], T=1$, $Nx = 128, Nt = 100$

### Burgers (IC fixed)
$\Omega = [0,1], T=1$, $Nx = 101, Nt = 101$. 


