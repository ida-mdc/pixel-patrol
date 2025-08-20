import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
from pixel_patrol.core.loader_interface import PixelPatrolLoader

logger = logging.getLogger(__name__)


class TraccuracyLoader(PixelPatrolLoader):

    @staticmethod
    def id() -> str:
        return "traccuracy"

    @property
    def reads_only_metadata(self) -> bool:
        return True


    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Recursively flattens a nested dictionary and sanitizes keys."""
        items = {}
        for k, v in d.items():
            sanitized_k = str(k).replace(' ', '_').replace('-', '_')
            new_key = parent_key + sep + sanitized_k if parent_key else sanitized_k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    def read_metadata_and_data(self, path: Path) -> Tuple[Dict[str, Any], Optional[da.Array]]:
        """
        Parses a traccuracy file, creating a detailed metadata record from all metric blocks.
        """
        logger.debug(f"Attempting to read '{path}' with TraccuracyLoader.")

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise IOError(f"Failed to read or parse JSON from '{path}': {e}")

        traccuracy_list = data.get("traccuracy")
        if not isinstance(traccuracy_list, list) or not traccuracy_list:
            raise ValueError(f"'{path}' is not a valid traccuracy file (missing or empty 'traccuracy' list).")

        metadata: Dict[str, Any] = {}

        # --- Process each metric block independently ---
        for metric_block in traccuracy_list:
            if not isinstance(metric_block, dict):
                continue

            metric_name = metric_block.get("metric", {}).get("name", "UnknownMetric")
            prefix = f"traccuracy_{metric_name}"

            # --- Extract and prefix all data for this block ---
            if version := metric_block.get('version'):
                metadata[f"{prefix}_version"] = version
            if gt := metric_block.get('gt'):
                metadata[f"{prefix}_gt"] = gt
            if pred := metric_block.get('pred'):
                metadata[f"{prefix}_pred"] = pred

            if matcher_info := metric_block.get('matcher'):
                for k, v in self._flatten_dict(matcher_info).items():
                    metadata[f"{prefix}_matcher_{k}"] = v

            if results_dict := metric_block.get("results"):
                for k, v in self._flatten_dict(results_dict).items():
                    metadata[f"{prefix}_results_{k}"] = v

            # Process metric parameters, excluding the name which is already used
            metric_params = {k: v for k, v in metric_block.get("metric", {}).items() if k != 'name'}
            if metric_params:
                for k, v in self._flatten_dict(metric_params).items():
                    metadata[f"{prefix}_metric_{k}"] = v

        logger.info(f"Successfully parsed precise traccuracy metadata from '{path}'.")
        return metadata, None

    def read_metadata(self, path: Path) -> Dict[str, Any]:
        """Reads and flattens metadata from a traccuracy JSON file."""
        try:
            metadata, _ = self.read_metadata_and_data(path)
            return metadata
        except (IOError, ValueError):
            raise

    def get_specification(self) -> Dict[str, Any]:
        """
        No static fields are defined, as all keys are dynamically generated
        based on the metric names found in the file.
        """
        return {}

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        """
        Defines precise patterns for dynamically generated columns.
        The order is important: more specific patterns must come first.
        """
        return [
            (r'^traccuracy_.*_metric_relax_skips_pred$', bool),
            (r'^traccuracy_.*_metric_relax_skips_gt$', bool),

            (r'^traccuracy_.*_metric_valid_match_types$', list[str]),

            (r'^traccuracy_.*_version$', str),
            (r'^traccuracy_.*_gt$', str),
            (r'^traccuracy_.*_pred$', str),

            (r'^traccuracy_.*_matcher_name$', str),
            (r'^traccuracy_.*_matcher_matching_type$', str),

            (r'^traccuracy_.*_results_.*$', float),
        ]