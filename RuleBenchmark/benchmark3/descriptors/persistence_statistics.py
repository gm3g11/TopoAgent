"""Persistence Statistics Tool - Benchmark3 fixed copy.

Fix: Corrected n_stats in empty-case handler.
Original had {'basic': 14, 'extended': 25, 'full': 35} but actual vector lengths are:
  basic=14, extended=14+7=21, full=14+7+10=31
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to use numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: identity decorator
    def jit(nopython=True, cache=True):
        def decorator(func):
            return func
        return decorator


class PersistenceStatisticsInput(BaseModel):
    """Input schema for PersistenceStatisticsTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    subset: str = Field(
        "extended", description="Statistics subset: 'basic' (14D), 'extended' (21D), or 'full' (31D)"
    )


class PersistenceStatisticsTool(BaseTool):
    """Compute comprehensive statistics from persistence diagrams.

    This tool extracts statistical features from birth times, death times,
    and persistence (lifespan) values. Different subsets provide different
    levels of detail:

    - basic (14 stats per dim): mean, std, median, IQR, min, max, count,
                                sum, q25, q75, skewness, kurtosis, range, entropy
    - extended (21 stats per dim): adds birth/death means+stds + 3 correlations
    - full (31 stats per dim): adds 10 advanced ratios and cross-statistics
    """

    name: str = "persistence_statistics"
    description: str = (
        "Compute comprehensive statistics from persistence diagrams. "
        "Extracts mean, std, median, min, max, percentiles, entropy, etc. "
        "Input: persistence data from compute_ph tool. "
        "Output: fixed-size vector (14D basic, 21D extended, 31D full per dim). "
        "Fast and interpretable features for ML classifiers."
    )
    args_schema: Type[BaseModel] = PersistenceStatisticsInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        subset: str = "extended",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute persistence statistics."""
        try:
            if subset not in ['basic', 'extended', 'full']:
                return {
                    "success": False,
                    "tool_name": self.name,
                    "error": f"Invalid subset: {subset}. Use 'basic', 'extended', or 'full'"
                }

            all_stats = {}
            all_vectors = {}

            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]

                # Extract birth, death, persistence arrays
                births, deaths, persistences = self._extract_arrays(pairs)

                # Compute statistics
                stats, vector = self._compute_stats(births, deaths, persistences, subset)

                all_stats[dim_key] = stats
                all_vectors[dim_key] = vector.tolist()

            # Combine vectors from all dimensions
            combined_vector = []
            for dim_key in sorted(all_vectors.keys()):
                combined_vector.extend(all_vectors[dim_key])

            return {
                "success": True,
                "tool_name": self.name,
                "statistics": all_stats,
                "vectors": all_vectors,
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "subset": subset,
                "interpretation": self._interpret(all_stats)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _extract_arrays(self, pairs: List[Dict]) -> tuple:
        """Extract birth, death, persistence arrays from pairs."""
        births = []
        deaths = []
        persistences = []

        for p in pairs:
            if not isinstance(p, dict) or "birth" not in p or "death" not in p:
                continue

            birth = p["birth"]
            death = p["death"]

            if not np.isfinite(death):
                continue  # Skip infinite persistence

            births.append(birth)
            deaths.append(death)
            persistences.append(death - birth)

        return (
            np.array(births, dtype=np.float64),
            np.array(deaths, dtype=np.float64),
            np.array(persistences, dtype=np.float64)
        )

    def _compute_stats(
        self,
        births: np.ndarray,
        deaths: np.ndarray,
        persistences: np.ndarray,
        subset: str
    ) -> tuple:
        """Compute statistics for given arrays."""
        stats = {}
        vector_parts = []

        # Handle empty arrays
        if len(persistences) == 0:
            # FIX: Corrected n_stats to match actual vector lengths
            # basic=14, extended=14+7=21, full=14+7+10=31
            n_stats = {'basic': 14, 'extended': 21, 'full': 31}[subset]
            return stats, np.zeros(n_stats)

        # Basic statistics (always computed)
        basic_stats = self._compute_basic_stats(persistences)
        stats['basic'] = basic_stats
        vector_parts.append(self._stats_to_vector(basic_stats))

        if subset in ['extended', 'full']:
            # Extended: per-type statistics
            birth_stats = self._compute_basic_stats(births)
            death_stats = self._compute_basic_stats(deaths)
            stats['birth'] = birth_stats
            stats['death'] = death_stats

            # Additional extended stats
            extended = {
                'birth_death_correlation': self._safe_corrcoef(births, deaths),
                'birth_persistence_correlation': self._safe_corrcoef(births, persistences),
                'death_persistence_correlation': self._safe_corrcoef(deaths, persistences),
            }
            stats['extended'] = extended

            vector_parts.append(np.array([
                birth_stats['mean'], birth_stats['std'],
                death_stats['mean'], death_stats['std'],
                extended['birth_death_correlation'],
                extended['birth_persistence_correlation'],
                extended['death_persistence_correlation'],
            ]))

        if subset == 'full':
            # Full: advanced ratios and cross-statistics
            full_stats = {
                'total_persistence': np.sum(persistences),
                'mean_midlife': np.mean((births + deaths) / 2) if len(births) > 0 else 0,
                'std_midlife': np.std((births + deaths) / 2) if len(births) > 0 else 0,
                'amplitude': np.max(deaths) - np.min(births) if len(births) > 0 else 0,
                'persistence_weighted_mean_birth': self._weighted_mean(births, persistences),
                'persistence_weighted_mean_death': self._weighted_mean(deaths, persistences),
                'persistence_squared_sum': np.sum(persistences ** 2),
                'persistence_cubed_sum': np.sum(persistences ** 3),
                'log_persistence_sum': np.sum(np.log(persistences + 1e-10)),
                'normalized_entropy': self._normalized_entropy(persistences),
            }
            stats['full'] = full_stats
            vector_parts.append(np.array(list(full_stats.values())))

        return stats, np.concatenate(vector_parts)

    def _compute_basic_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics for an array."""
        if len(values) == 0:
            return {k: 0.0 for k in [
                'count', 'mean', 'std', 'median', 'min', 'max',
                'iqr', 'q25', 'q75', 'sum', 'skewness', 'kurtosis',
                'range', 'entropy'
            ]}

        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)

        return {
            'count': float(len(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'iqr': float(q75 - q25),
            'q25': float(q25),
            'q75': float(q75),
            'sum': float(np.sum(values)),
            'skewness': float(self._skewness(values)),
            'kurtosis': float(self._kurtosis(values)),
            'range': float(np.max(values) - np.min(values)),
            'entropy': float(self._histogram_entropy(values)),
        }

    def _stats_to_vector(self, stats: Dict[str, float]) -> np.ndarray:
        """Convert stats dict to ordered vector."""
        keys = ['count', 'mean', 'std', 'median', 'min', 'max',
                'iqr', 'q25', 'q75', 'sum', 'skewness', 'kurtosis',
                'range', 'entropy']
        return np.array([stats[k] for k in keys])

    @staticmethod
    def _skewness(values: np.ndarray) -> float:
        """Compute skewness."""
        if len(values) < 3:
            return 0.0
        n = len(values)
        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-10:
            return 0.0
        return (n / ((n - 1) * (n - 2))) * np.sum(((values - mean) / std) ** 3)

    @staticmethod
    def _kurtosis(values: np.ndarray) -> float:
        """Compute kurtosis (excess)."""
        if len(values) < 4:
            return 0.0
        n = len(values)
        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-10:
            return 0.0
        m4 = np.mean((values - mean) ** 4)
        return m4 / (std ** 4) - 3

    @staticmethod
    def _histogram_entropy(values: np.ndarray, bins: int = 20) -> float:
        """Compute histogram entropy."""
        if len(values) < 2:
            return 0.0
        hist, _ = np.histogram(values, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        return -np.sum(hist * np.log(hist + 1e-10))

    @staticmethod
    def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
        """Safely compute correlation coefficient."""
        if len(x) < 2:
            return 0.0
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted mean."""
        if len(values) == 0 or np.sum(weights) < 1e-10:
            return 0.0
        return float(np.sum(values * weights) / np.sum(weights))

    @staticmethod
    def _normalized_entropy(values: np.ndarray) -> float:
        """Compute normalized entropy (0 to 1)."""
        if len(values) < 2:
            return 0.0
        total = np.sum(values)
        if total < 1e-10:
            return 0.0
        p = values / total
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(values))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _interpret(self, all_stats: Dict) -> str:
        """Generate interpretation of statistics."""
        parts = []

        for dim_key in sorted(all_stats.keys()):
            stats = all_stats[dim_key]
            if 'basic' in stats:
                basic = stats['basic']
                parts.append(
                    f"{dim_key}: {int(basic['count'])} features, "
                    f"mean persistence={basic['mean']:.4f}"
                )

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
