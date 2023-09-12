

import torch
import re

def _parse_expression(expression, parameter_list):
    for parameter in parameter_list:
        expression = expression.replace(parameter, f'x[{parameter_list.index(parameter)}]')
    return expression

def create_callable_function(constraint_string, parameter_list) -> callable:
    '''
    # Example usage:
    parameter_list = ["ParameterA", "ParameterB", "ParameterC", "ParameterD"]
    constraint_string = "ParameterA + ParameterD <= 10"

    equation_function = create_callable_function(constraint_string, parameter_list)

    # Now you can call the equation function with the x array
    x_values = [3, 200, 500, 8]
    result = equation_function(x_values)
    print("Result:", result)
    '''
    # Validate the constraint string format for <= or >= operators
    regex = r"\s*([\w\s+\-]+)\s*(<=|>=)\s*(\d+(\.\d+)?)\s*"
    match = re.match(regex, constraint_string)
    if not match:
        raise ValueError("Invalid constraint string format. Only >= and <= operators are allowed.")

    # Extract the constraint parameters and operator from the matched groups
    constraint_expr, _, value, _ = match.groups()

    # Replace the parameter names with their corresponding values in the constraint expression
    parsed_expr = _parse_expression(constraint_expr, parameter_list)

    # Build the equation function for the constraint
    def equation(x):
        if torch.is_tensor(x):
            return float(value) - eval(parsed_expr, {"x": x.cpu().numpy()})
        return float(value) - eval(parsed_expr, {"x": x})

    return equation

def create_callable_function_bool(constraint_string, parameter_list) -> callable:
    '''
    # Example usage:
    parameter_list = ["ParameterA", "ParameterB", "ParameterC", "ParameterD"]
    constraint_string = "ParameterA + ParameterD <= 10"

    equation_function = create_callable_function(constraint_string, parameter_list)

    # Now you can call the equation function with the x array
    x_values = [3, 200, 500, 8]
    result = equation_function(x_values) # Return Boolean
    print("Result:", result)
    '''
    # Validate the constraint string format for <= or >= operators
    regex = r"\s*([\w\s+\-]+)\s*(<=|>=)\s*(\d+(\.\d+)?)\s*"
    match = re.match(regex, constraint_string)
    if not match:
        raise ValueError("Invalid constraint string format. Only >= and <= operators are allowed.")

    # Extract the constraint parameters and operator from the matched groups
    constraint_expr, _, value, _ = match.groups()

    # Replace the parameter names with their corresponding values in the constraint expression
    parsed_expr = _parse_expression(constraint_expr, parameter_list)

    # Build the equation function for the constraint
    def equation(x):
        if torch.is_tensor(x):
            return  ( float(value) - eval(parsed_expr, {"x": x.cpu().numpy()}) ) >= 0
        return  ( float(value) - eval(parsed_expr, {"x": x}) ) >= 0

    return equation

def format_outcome_constraints_botorch(constraints, objectives) -> list:
    '''
    in most cases:
    objectives = use_case_runner.objectives 
    '''
    constraints_formatted = None
    if not constraints:
        return constraints_formatted

    def get_objectives(input_string):
        pattern = r"([^\s+<=>-]+)"
        objectives = re.findall(pattern, input_string)
        return [p for p in objectives if not any(c.isalpha() for c in p)]

    def get_index_list(objectives):
        return [objectives.index(p.replace("'", "")) for p in objectives]

    def get_botorch_constraint_data(input_string):
        pattern = r"([^\s+]+)\s*([<==]+)\s*([^\s]+)"
        matches = re.findall(pattern, input_string)
        if matches:
            obj_idx = get_index_list(get_objectives(input_string))
            return [(torch.tensor([int(i in obj_idx) for i in range(len(objectives))], dtype=torch.double),
                     torch.tensor(float(value), dtype=torch.double)) for _, operator, value in matches if operator in ("<=", "==")]

    constraints_formatted = get_botorch_constraint_data(constraints)
    return constraints_formatted