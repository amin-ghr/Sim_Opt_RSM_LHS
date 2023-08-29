# Assembly Line Optimization using Response Surface Methodology (RSM)

This repository contains a Python implementation of an assembly line optimization problem using Response Surface Methodology (RSM) and Latin Hypercube Sampling (LHS). The goal is to minimize costs while improving production processes by tuning assembly line capacity and the number of workers.

## Problem Statement

The assembly line aims to optimize the number of workers and assembly line capacity. Costs are influenced by worker wages, production rate, and worker efficiency. A simulation model captures the assembly process, and a quadratic response surface model is used to optimize the variables.

## Features

- Sequential assembly line simulation model
- Design of Experiments (DoE) using Latin Hypercube Sampling (LHS)
- Quadratic response surface model fitting
- Optimization using Basin Hopping algorithm
- Contour plots to visualize the response surface

## Usage

1. Install required packages: `pip install simpy numpy matplotlib scipy pyDOE2`
2. Run `assembly_line_simulation_optimization.py` to optimize the assembly line.

## Results

Optimized input variables are found using RSM, leading to improved assembly line efficiency and minimized costs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
