import re
def camel_to_snake(name:str):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def snake_to_camel(name:str, first_upper = False):
    components = name.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    if first_upper == False:
        return components[0] + ''.join(x.title() for x in components[1:])
    else:
        return components[0][0].upper()+ components[0][1:]+ ''.join(x.title() for x in components[1:])

def calc_num_arms_from_name(arm_name: str, init_trials: int, batch_size: int):
    current_trial, current_arm = arm_name.split("_")
    return init_trials + ((int(current_trial) - init_trials) * batch_size) + int(current_arm) 


