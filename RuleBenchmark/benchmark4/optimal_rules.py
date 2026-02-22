"""
Optimal Rules Loader for Benchmark4.

Loads exp4_final_recommendations.json and provides a clean API
for retrieving optimal dimension (D*) and parameters for each
(descriptor, object_type) pair.

Usage:
    rules = OptimalRules()
    params = rules.get_optimal_params('persistence_image', 'discrete_cells')
    # -> {'resolution': 10, 'sigma': 0.6, 'weight_function': 'squared'}
    dim = rules.get_expected_dim('persistence_image', 'discrete_cells')
    # -> 200
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from RuleBenchmark.benchmark4.config import EXP4_RULES_PATH


class OptimalRules:
    """Load and query exp4 optimal rules."""

    def __init__(self, rules_path: Optional[Path] = None):
        path = rules_path or EXP4_RULES_PATH
        with open(path, 'r') as f:
            self._data = json.load(f)
        self._dim_rules = self._data['dimension_rules']
        self._param_rules = self._data['parameter_rules']

    def get_dimension_params(self, descriptor: str, object_type: str) -> Dict[str, Any]:
        """Get the optimal dimension control parameter and expected dim.

        Returns dict with the control parameter value(s) and 'dim'.
        Falls back to default if object_type not found.
        """
        if descriptor not in self._dim_rules:
            raise ValueError(f"Unknown descriptor: {descriptor}")

        rule = self._dim_rules[descriptor]

        # persistence_statistics has no per-type variation
        if 'rules_all' in rule:
            return dict(rule['rules_all'])

        rules = rule.get('rules', {})
        if object_type in rules:
            result = dict(rules[object_type])
        else:
            result = dict(rule.get('default', {}))

        # Remove metadata keys
        result.pop('accuracy', None)
        result.pop('note', None)
        return result

    def get_expected_dim(self, descriptor: str, object_type: str,
                         per_channel: bool = False) -> int:
        """Get expected feature dimension D*.

        Args:
            descriptor: Descriptor name
            object_type: Object type
            per_channel: If True, returns 3 * D* for RGB per-channel mode

        Returns:
            Expected feature dimension
        """
        params = self.get_dimension_params(descriptor, object_type)
        dim = params.get('dim', 0)
        if per_channel:
            dim *= 3
        return dim

    def get_tuned_params(self, descriptor: str, object_type: str) -> Dict[str, Any]:
        """Get tuned parameters (sigma, weight_fn, etc.) from parameter search.

        Returns empty dict if no parameter tuning was done for this descriptor.
        """
        if descriptor not in self._param_rules:
            return {}

        rule = self._param_rules[descriptor]
        per_type = rule.get('per_object_type', {})

        if object_type in per_type:
            result = dict(per_type[object_type])
        else:
            result = dict(rule.get('best_params', {}))

        # Remove metadata
        result.pop('accuracy', None)
        result.pop('note', None)
        return result

    def get_optimal_params(self, descriptor: str, object_type: str) -> Dict[str, Any]:
        """Get ALL optimal parameters for a (descriptor, object_type) pair.

        Merges dimension params and tuned params into a single dict.
        """
        params = self.get_dimension_params(descriptor, object_type)
        tuned = self.get_tuned_params(descriptor, object_type)
        params.update(tuned)
        return params

    def get_descriptor_config(self, descriptor: str, object_type: str,
                              color_mode: str = 'grayscale') -> Dict[str, Any]:
        """Get complete descriptor configuration for evaluation.

        Returns dict with:
            params: all optimal params
            dim_per_channel: D* per channel
            total_dim: total feature dimension (3*D* for per_channel)
            color_mode: 'grayscale' or 'per_channel'
        """
        params = self.get_optimal_params(descriptor, object_type)
        dim = params.pop('dim', 0)
        per_channel = (color_mode == 'per_channel')

        return {
            'params': params,
            'dim_per_channel': dim,
            'total_dim': dim * 3 if per_channel else dim,
            'color_mode': color_mode,
        }


# Module-level singleton for convenience
_rules_instance = None

def get_rules() -> OptimalRules:
    """Get the singleton OptimalRules instance."""
    global _rules_instance
    if _rules_instance is None:
        _rules_instance = OptimalRules()
    return _rules_instance


if __name__ == '__main__':
    rules = OptimalRules()

    # Print all optimal configs for BloodMNIST (discrete_cells)
    print("=== BloodMNIST (discrete_cells) ===")
    from RuleBenchmark.benchmark4.config import ALL_DESCRIPTORS
    for desc in ALL_DESCRIPTORS:
        cfg = rules.get_descriptor_config(desc, 'discrete_cells', 'per_channel')
        print(f"  {desc:35s}  dim={cfg['total_dim']:5d}  params={cfg['params']}")

    print("\n=== OrganAMNIST (organ_shape, grayscale) ===")
    for desc in ALL_DESCRIPTORS:
        cfg = rules.get_descriptor_config(desc, 'organ_shape', 'grayscale')
        print(f"  {desc:35s}  dim={cfg['total_dim']:5d}  params={cfg['params']}")
