from ..type import StateDict
from .arithmetic import *


def check_parameterNamesMatch(checkpoints: List[StateDict]) -> None:
    """
    Checks that the parameter names of the given checkpoints match.

    Args:
        checkpoints (List[Dict[str, float]]): A list of checkpoints, where each checkpoint is a dictionary of parameter names and their corresponding values.

    Raises:
        ValueError: If the number of checkpoints is less than 2 or if the parameter names of any two checkpoints differ.

    """
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )
