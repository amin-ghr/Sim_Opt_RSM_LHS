import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from pyDOE2 import lhs

# Define the sequential assembly line simulation model
class AssemblyLineSimulationModel:
    def __init__(self, input_variables):
        self.input_variables = input_variables
        self.env = simpy.Environment()
        self.total_cost = 0
        self.production_rate = input_variables['production_rate']
        self.worker_rate = input_variables['worker_rate']
        self.num_products = input_variables['num_products']
    
    def run_simulation(self):
        assembly_line_capacity = self.input_variables['assembly_line_capacity']
        worker_efficiency = self.input_variables['worker_efficiency']
        
        # Define resources
        assembly_line = simpy.Resource(self.env, capacity=assembly_line_capacity)
        
        # Update the cost based on the line capacity
        self.total_cost -= assembly_line_capacity * 100 # $10 per unit increase in line capacity
        
        # Start the assembly line processes
        for i in range(self.num_products):
            # Request a worker from the assembly line
            with assembly_line.request() as request:
                yield request
                
                # Perform the assembly process
                processing_time = 1 / (self.production_rate * worker_efficiency)
                yield self.env.timeout(processing_time)
                
                # Update the cost based on the processing time and worker rate
                self.total_cost -= processing_time * self.worker_rate * (10 * worker_efficiency) # $20 per hour is the worker wage depending on the efficiency

                # Update the cost based on the line capacity
                production_rate = 1 / (self.production_rate * worker_efficiency * self.worker_rate)

                # add profit for production
                self.total_cost += 200 * production_rate
        
    def get_total_cost(self):
        return self.total_cost

# Define the objective function for optimization
def objective(input_variables):
    # Create an instance of the assembly line simulation model with the current input variables
    assembly_line_model = AssemblyLineSimulationModel(input_variables)
    
    # Run the simulation
    assembly_line_model.env.process(assembly_line_model.run_simulation())
    assembly_line_model.env.run()
    
    # Get the total cost as the output response to minimize
    total_cost = assembly_line_model.get_total_cost()
    
    return total_cost

# Perform Design of Experiments (DoE) using Latin Hypercube Sampling (LHS)
def perform_doe(num_samples):
    # Define the ranges for the input variables
    production_rate_range = (50, 200)  # Range for the production rate
    worker_rate_range = (5, 15)  # Range for the worker rate
    assembly_line_capacity_range = (50, 200)  # Range for the assembly line capacity
    worker_efficiency_range = (0.8, 1.2)  # Range for worker efficiency
    
    # Generate Latin Hypercube Sampling (LHS) samples
    samples = lhs(4, samples=num_samples)
    print('length of the samples: ', len(samples))
    
    # Scale the samples to the defined ranges
    production_rate_values = production_rate_range[0] + samples[:, 0] * (production_rate_range[1] - production_rate_range[0])
    worker_rate_values = worker_rate_range[0] + samples[:, 1] * (worker_rate_range[1] - worker_rate_range[0])
    assembly_line_capacity_values = assembly_line_capacity_range[0] + samples[:, 2] * (assembly_line_capacity_range[1] - assembly_line_capacity_range[0])
    worker_efficiency_values = worker_efficiency_range[0] + samples[:, 3] * (worker_efficiency_range[1] - worker_efficiency_range[0])
    
    # Create a list of input variable settings based on the LHS samples
    input_variables_list = []
    for i in range(num_samples):
        input_variables = {
            'production_rate': production_rate_values[i],
            'worker_rate': worker_rate_values[i],
            'assembly_line_capacity': assembly_line_capacity_values[i],
            'worker_efficiency': worker_efficiency_values[i],
            'num_products': 200  # Arbitrary value for the number of products
        }
        input_variables_list.append(input_variables)
    
    return input_variables_list

# Fit a quadratic response surface model to the DoE data
def fit_quadratic_model(input_variables_list, costs):
    # Extract the input variable values
    production_rate_values = np.array([input_variables['production_rate'] for input_variables in input_variables_list])
    worker_rate_values = np.array([input_variables['worker_rate'] for input_variables in input_variables_list])
    assembly_line_capacity_values = np.array([input_variables['assembly_line_capacity'] for input_variables in input_variables_list])
    worker_efficiency_values = np.array([input_variables['worker_efficiency'] for input_variables in input_variables_list])
    
    # Create the design matrix with quadratic terms
    X = np.column_stack([
        np.ones(len(input_variables_list)), 
        production_rate_values,
        worker_rate_values,
        assembly_line_capacity_values,
        worker_efficiency_values,
        production_rate_values ** 2,
        worker_rate_values ** 2,
        assembly_line_capacity_values ** 2,
        worker_efficiency_values ** 2,
        production_rate_values * worker_rate_values,
        production_rate_values * assembly_line_capacity_values,
        production_rate_values * worker_efficiency_values,
        worker_rate_values * assembly_line_capacity_values,
        worker_rate_values * worker_efficiency_values,
        assembly_line_capacity_values * worker_efficiency_values,
    ])
    
    # Fit a quadratic model using least squares
    coeffs = np.linalg.lstsq(X, costs, rcond=None)[0]
    
    return coeffs

# Use RSM to optimize the sequential assembly line
def optimize_assembly_line():
    # Perform Design of Experiments (DoE) using Latin Hypercube Sampling (LHS)
    num_samples = 100
    input_variables_list = perform_doe(num_samples)
    
    # Evaluate the costs for the DoE samples
    doe_costs = []
    for input_variables in input_variables_list:
        cost = objective(input_variables)
        doe_costs.append(cost)
    
    # Fit a quadratic response surface model to the DoE data
    coeffs = fit_quadratic_model(input_variables_list, doe_costs)
    
    # Define the objective function for scipy.optimize using the quadratic model
    def scipy_objective(x):
        production_rate, worker_rate, assembly_line_capacity, worker_efficiency = x

        return coeffs[0] + \
               coeffs[1] * production_rate + coeffs[2] * worker_rate + coeffs[3] * assembly_line_capacity + \
               coeffs[4] * worker_efficiency + coeffs[5] * production_rate ** 2 + coeffs[6] * worker_rate ** 2 + \
               coeffs[7] * assembly_line_capacity ** 2 + coeffs[8] * worker_efficiency ** 2 + \
               coeffs[9] * production_rate * worker_rate + coeffs[10] * production_rate * assembly_line_capacity + \
               coeffs[11] * production_rate * worker_efficiency + coeffs[12] * worker_rate * assembly_line_capacity + \
               coeffs[13] * worker_rate * worker_efficiency + coeffs[14] * assembly_line_capacity * worker_efficiency
    
    # Define the bounds for the input variables
    bounds = [(50, 200), (5, 15), (50, 200), (0.8, 1.2)]
    
    # Perform optimization using Basin Hopping algorithm from scipy.optimize
    result = basinhopping(scipy_objective, x0=[100, 10, 100, 1], minimizer_kwargs={'bounds': bounds})
    
    # Extract the optimized input variable settings and total cost
    optimized_input_variables = {
        'production_rate': result.x[0],
        'worker_rate': result.x[1],
        'assembly_line_capacity': result.x[2],
        'worker_efficiency': result.x[3],
        'num_products': 100  # Arbitrary value for the number of products
    }
    optimized_total_cost = result.fun
    
    # Print the optimized input variable settings and total cost
    print("Optimized Input Variable Settings:")
    for name, value in optimized_input_variables.items():
        print(f"{name}: {value}")
    
    print("Optimized Total Cost: ", optimized_total_cost)
    
    # Generate the contour plot
    plot_contour(optimized_input_variables, input_variables_list, doe_costs, coeffs)

# Function to generate and display a contour plot
def plot_contour(optimized_input_variables, input_variables_list, doe_costs, coeffs):
    # Extract the input variable values
    production_rate_values = np.array([input_variables['production_rate'] for input_variables in input_variables_list])
    worker_rate_values = np.array([input_variables['worker_rate'] for input_variables in input_variables_list])
    
    # Create a meshgrid for contour plot
    resolution = 100
    production_rate_grid = np.linspace(50, 200, resolution)
    worker_rate_grid = np.linspace(5, 15, resolution)
    X, Y = np.meshgrid(production_rate_grid, worker_rate_grid)
    
    # Evaluate the total costs for each combination of production_rate and worker_rate using the quadratic model
    Z = coeffs[0] + \
        coeffs[1] * X + coeffs[2] * Y + coeffs[3] * optimized_input_variables['assembly_line_capacity'] + \
        coeffs[4] * optimized_input_variables['worker_efficiency'] + \
        coeffs[5] * X ** 2 + coeffs[6] * Y ** 2 + \
        coeffs[7] * optimized_input_variables['assembly_line_capacity'] ** 2 + \
        coeffs[8] * optimized_input_variables['worker_efficiency'] ** 2 + \
        coeffs[9] * X * Y + coeffs[10] * X * optimized_input_variables['assembly_line_capacity'] + \
        coeffs[11] * X * optimized_input_variables['worker_efficiency'] + \
        coeffs[12] * Y * optimized_input_variables['assembly_line_capacity'] + \
        coeffs[13] * Y * optimized_input_variables['worker_efficiency'] + \
        coeffs[14] * optimized_input_variables['assembly_line_capacity'] * optimized_input_variables['worker_efficiency']
    
    # Create the contour plot
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Total Cost')
    plt.scatter(production_rate_values, worker_rate_values, c=doe_costs, cmap='jet', label='DoE Samples')
    plt.scatter(optimized_input_variables['production_rate'], optimized_input_variables['worker_rate'],
                c='red', marker='x', label='Optimized Point')
    plt.xlabel('Production Rate')
    plt.ylabel('Worker Rate')
    plt.title('Response Surface and Design of Experiments')
    plt.legend()
    plt.show()

# Call the optimize_assembly_line() function to start the optimization process
optimize_assembly_line()
