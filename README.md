# Accurate Boundary Condition for Moving Least Squares Material Point Method using Augmented Grid Points

![teaser](https://github.com/nobuyuki83/accurate_bc_for_mls_mpm/blob/images/teaser.png)


- [Supplemental video on YouTube](https://youtu.be/Rcp94v6mU3w)

# Publication

:::note
Riku Toyota and Nobuyuki Umetani, 2024, 
"Accurate Boundary Condition for Moving Least Squares 
Material Point Method using Augmented Grid Points"
Eurographics 2024 Short Paper Program
:::



## Abstract
This paper introduces an accurate boundary-handling method for 
the moving least squares (MLS) material point method (MPM), 
which is a popular scheme for robustly simulating deformable objects and 
fluids using a hybrid of particle and grid representations coupled 
via MLS interpolation. Despite its versatility with different materials, 
traditional MPM suffers from undesirable artifacts around wall boundaries,
for example, particles pass through the walls and accumulate. 
To address these issues, we present a technique inspired by a line handler
for MLS-based image manipulation. Specifically, we augment the grid 
by adding points along the wall boundary to numerically compute 
the integration of the MLS weight. These additional points act 
as background grid points, improving the accuracy of the MLS 
interpolation around the boundary, albeit with a marginal increase 
in computational cost. In particular, our technique makes 
the velocity perpendicular to the wall nearly zero, 
preventing particles from passing through the wall.
We compare the boundary behavior of 2D simulation against that of 
naive approach.

| naive      | ours     |
|------------|----------|
| ![naive](https://github.com/nobuyuki83/accurate_bc_for_mls_mpm/blob/images/naive_nonslip.gif) | ![our](https://github.com/nobuyuki83/accurate_bc_for_mls_mpm/blob/images/ours_nonslip.gif) |
| ![naive](https://github.com/nobuyuki83/accurate_bc_for_mls_mpm/blob/images/naive_nonslip_sphere.gif) | ![our](https://github.com/nobuyuki83/accurate_bc_for_mls_mpm/blob/images/ours_nonslip_sphere.gif) |


# How to build

The demos are written in `Rust`. Please [install rust development environment](https://www.rust-lang.org/learn/get-started) if you do not have rust in your computer.    

The figures can be reproduced with the following commands.
```bash
# to reproduce Fig.1-top, Fig.1-bottom, and Fig.5-top
# the outputs are in the target folder 
cargo run --example fluid --release

# to reproduce Fig.5-middle and Fig.5-bottom
# the outputs are in the target folder
cargo run --example solid --release

# to reproduce Fig.4
# the outputs are in the target folder
cargo run --example sdf --release 
```