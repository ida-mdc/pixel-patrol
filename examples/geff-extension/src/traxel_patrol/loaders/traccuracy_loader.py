import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

from pixel_patrol_base.core.artifact import Artifact

logger = logging.getLogger(__name__)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """Recursively flattens a nested dictionary and sanitizes keys."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        sanitized_k = str(k).replace(" ", "_").replace("-", "_")
        new_key = parent_key + sep + sanitized_k if parent_key else sanitized_k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


class TraccuracyLoader:
    """
    Artifact-first loader for traccuracy JSON files.

    - Single entrypoint: `load(source) -> Artifact`
    - Returns an Artifact carrying all flattened traccuracy metrics in `meta`
      (no image data; processors that require XY/caps will be skipped).
    """

    NAME = "traccuracy"

    SUPPORTED_EXTENSIONS: Set[str, Any] = ["json"]

    # Declarative schema contributed by this loader (replaces get_specification / get_dynamic_specification_patterns)
    OUTPUT_SCHEMA: Dict[str, Any] = {}
    OUTPUT_SCHEMA_PATTERNS: List[Tuple[str, Any]] = [
        (r"^traccuracy_.*_metric_relax_skips_pred$", bool),
        (r"^traccuracy_.*_metric_relax_skips_gt$", bool),
        (r"^traccuracy_.*_metric_valid_match_types$", list),   # or encode to JSON str if you prefer non-Object dtype
        (r"^traccuracy_.*_version$", str),
        (r"^traccuracy_.*_gt$", str),
        (r"^traccuracy_.*_pred$", str),
        (r"^traccuracy_.*_matcher_name$", str),
        (r"^traccuracy_.*_matcher_matching_type$", str),
        (r"^traccuracy_.*_results_.*$", float),
    ]

    def load(self, source: str) -> Artifact:
        """
        Parses a traccuracy file, creating a detailed metadata record from all metric blocks.
        """
        path = Path(source)
        logger.debug(f"Attempting to read '{path}' with TraccuracyLoader.")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise IOError(f"Failed to read or parse JSON from '{path}': {e}")

        traccuracy_list = data.get("traccuracy")
        if not isinstance(traccuracy_list, list) or not traccuracy_list:
            raise ValueError(f"'{path}' is not a valid traccuracy file (missing or empty 'traccuracy' list).")

        metadata: Dict[str, Any] = {}

        # Process each metric block independently
        for metric_block in traccuracy_list:
            if not isinstance(metric_block, dict):
                continue

            metric_name = metric_block.get("metric", {}).get("name", "UnknownMetric")
            prefix = f"traccuracy_{metric_name}"

            # Basic fields
            if (version := metric_block.get("version")) is not None:
                metadata[f"{prefix}_version"] = version
            if (gt := metric_block.get("gt")) is not None:
                metadata[f"{prefix}_gt"] = gt
            if (pred := metric_block.get("pred")) is not None:
                metadata[f"{prefix}_pred"] = pred

            # Matcher block
            if matcher_info := metric_block.get("matcher"):
                for k, v in _flatten_dict(matcher_info).items():
                    metadata[f"{prefix}_matcher_{k}"] = v

            # Results block
            if results_dict := metric_block.get("results"):
                for k, v in _flatten_dict(results_dict).items():
                    metadata[f"{prefix}_results_{k}"] = v

            # Metric parameters (excluding the name)
            metric_params = {k: v for k, v in metric_block.get("metric", {}).items() if k != "name"}
            if metric_params:
                for k, v in _flatten_dict(metric_params).items():
                    metadata[f"{prefix}_metric_{k}"] = v

        logger.info(f"Successfully parsed traccuracy metadata from '{path}'.")

        # No spatial axes/capabilities for pure metrics; processors needing XY/caps won't run on this artifact.
        axes: Set[str] = set()
        caps: Set[str] = set()

        return Artifact(
            data=None,        # payload not needed; all info is in meta
            axes=axes,
            kind="any",       # or define a new kind like "metrics" in your Artifact type if you prefer
            meta=metadata,
            capabilities=caps,
        )
