"""Color Mode Selection Skill.

Determines whether to use grayscale or per_channel (RGB) mode
based on image properties or dataset metadata.
"""

from typing import Optional

from .rules_data import DATASET_COLOR_MODE, COLOR_RULES


def select_from_dataset(dataset_name: str) -> Optional[str]:
    """Select color mode from known dataset metadata.

    Args:
        dataset_name: Dataset name

    Returns:
        'grayscale' or 'per_channel', or None if unknown
    """
    # Exact match
    if dataset_name in DATASET_COLOR_MODE:
        return DATASET_COLOR_MODE[dataset_name]

    # Case-insensitive
    name_lower = dataset_name.lower()
    for ds, mode in DATASET_COLOR_MODE.items():
        if ds.lower() == name_lower:
            return mode

    return None


def select_from_channels(n_channels: int) -> str:
    """Select color mode based on number of image channels.

    Args:
        n_channels: Number of image channels (1 or 3)

    Returns:
        'grayscale' or 'per_channel'
    """
    if n_channels >= 3:
        return "per_channel"
    return "grayscale"


def select(
    dataset_name: Optional[str] = None,
    n_channels: Optional[int] = None,
    image_path: Optional[str] = None,
) -> str:
    """Select color mode using all available signals.

    Priority:
    1. Known dataset lookup
    2. Channel count
    3. Image path hints
    4. Default to 'grayscale'

    Args:
        dataset_name: Optional dataset name
        n_channels: Optional number of image channels
        image_path: Optional image file path

    Returns:
        'grayscale' or 'per_channel'
    """
    # Known dataset
    if dataset_name:
        mode = select_from_dataset(dataset_name)
        if mode:
            return mode

    # Channel count
    if n_channels is not None:
        return select_from_channels(n_channels)

    # Image path: check for dataset names
    if image_path:
        path_lower = image_path.lower()
        for ds_name, mode in DATASET_COLOR_MODE.items():
            if ds_name.lower() in path_lower:
                return mode

    return "grayscale"
