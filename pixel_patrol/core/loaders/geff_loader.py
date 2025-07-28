import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional

import dask.array as da
import geff
import networkx as nx
import numpy as np
import zarr
from pixel_patrol.core.loader_interface import PixelPatrolLoader

logger = logging.getLogger(__name__)


def _calculate_global_graph_metrics(nx_graph: nx.DiGraph) -> Dict[str, Any]:
    """Computes graph-wide summary metrics."""
    return {
        "geff_num_nodes": nx_graph.number_of_nodes(),
        "geff_num_edges": nx_graph.number_of_edges(),
        "geff_num_lineages": len(list(nx.weakly_connected_components(nx_graph))),
        "geff_num_divisions": sum(1 for _, degree in nx_graph.out_degree() if degree > 1),
        "geff_num_terminations": sum(1 for _, degree in nx_graph.out_degree() if degree == 0),
    }


def _summarize_numerical_attributes(graph_view: Any, entity_type: str) -> Dict[str, Any]:
    """Dynamically finds and summarizes all numerical attributes for nodes or edges."""
    attr_values = defaultdict(list)
    for _, data in graph_view:
        for key, value in data.items():
            if isinstance(value, (int, float)):
                attr_values[key].append(value)

    summary_stats = {}
    for attr_name, values in attr_values.items():
        if not values:
            continue
        values_np = np.array(values)
        prefix = f"{entity_type}_attr_{attr_name}"
        summary_stats[f"geff_mean_{prefix}"] = np.mean(values_np)
        summary_stats[f"geff_std_{prefix}"] = np.std(values_np)
        summary_stats[f"geff_min_{prefix}"] = np.min(values_np)
        summary_stats[f"geff_max_{prefix}"] = np.max(values_np)

    return summary_stats

def _summarize_numerical_attributes_edges(graph_view: Any, entity_type: str) -> Dict[str, Any]:
    """Dynamically finds and summarizes all numerical attributes for nodes or edges."""
    attr_values = defaultdict(list)
    for _, _, data in graph_view:
        for key, value in data.items():
            if isinstance(value, (int, float)):
                attr_values[key].append(value)

    summary_stats = {}
    for attr_name, values in attr_values.items():
        if not values:
            continue
        values_np = np.array(values)
        prefix = f"{entity_type}_attr_{attr_name}"
        summary_stats[f"geff_mean_{prefix}"] = np.mean(values_np)
        summary_stats[f"geff_std_{prefix}"] = np.std(values_np)
        summary_stats[f"geff_min_{prefix}"] = np.min(values_np)
        summary_stats[f"geff_max_{prefix}"] = np.max(values_np)

    return summary_stats


def _calculate_timesliced_metrics(nx_graph: nx.DiGraph, geff_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates object counts, events, and attribute summaries per timepoint."""
    time_attr_name = next((ax['name'] for ax in geff_spec.get('axes', []) if ax.get('type') == 'time'), None)
    if not time_attr_name:
        return {}

    nodes_by_time = defaultdict(list)
    for node_id, data in nx_graph.nodes(data=True):
        if time_attr_name in data:
            nodes_by_time[data[time_attr_name]].append((node_id, data))

    timesliced_metrics = {}
    for t, nodes_at_t in sorted(nodes_by_time.items()):
        prefix = f"_T{t}"
        timesliced_metrics[f"geff_num_nodes{prefix}"] = len(nodes_at_t)

        out_degrees = [nx_graph.out_degree(n_id) for n_id, _ in nodes_at_t]
        timesliced_metrics[f"geff_num_divisions{prefix}"] = sum(1 for d in out_degrees if d > 1)
        timesliced_metrics[f"geff_num_terminations{prefix}"] = sum(1 for d in out_degrees if d == 0)

        # Dynamically summarize all numerical attributes for this time slice
        node_data_at_t = (data for _, data in nodes_at_t)
        slice_attr_stats = _summarize_numerical_attributes(zip(range(len(nodes_at_t)), node_data_at_t), "node")

        # Rename keys to be time-specific
        for key, value in slice_attr_stats.items():
            new_key = key.replace("geff_node_attr", f"{prefix}_attr")
            timesliced_metrics[new_key] = value

    return timesliced_metrics


class GeffLoader(PixelPatrolLoader):
    @property
    def reads_only_metadata(self) -> bool:
        return True

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        items = {}
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key and not k.startswith(parent_key) else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(self._flatten_dict(item, f"{new_key}_{i}", sep=sep))
                    else:
                        items[f"{new_key}_{i}"] = item
            else:
                items[new_key] = v
        return items

    def read_metadata_and_data(self, path: Path) -> Tuple[Dict[str, Any], Optional[da.Array]]:
        logger.debug(f"Attempting to read '{path}' with GeffLoader.")

        try:
            nx_graph, geff_meta_obj = geff.read_nx(path)
            geff_spec = geff_meta_obj.model_dump() if hasattr(geff_meta_obj, 'model_dump') else geff_meta_obj.dict()
        except Exception as e:
            raise IOError(f"Failed to read '{path}' as a GEFF file using the API. Error: {e}")

        metadata: Dict[str, Any] = {}

        # Add all raw, flattened GEFF spec metadata
        metadata.update(self._flatten_dict(geff_spec, parent_key='geff'))

        # Add computed metrics by calling helper functions
        metadata.update(_calculate_global_graph_metrics(nx_graph))
        metadata.update(_summarize_numerical_attributes(nx_graph.nodes(data=True), "node"))
        metadata.update(_summarize_numerical_attributes_edges(nx_graph.edges(data=True), "edge"))
        metadata.update(_calculate_timesliced_metrics(nx_graph, geff_spec))

        logger.info(f"Successfully parsed GEFF metadata from '{path}'.")
        return metadata, None

    def read_metadata(self, path: Path) -> Dict[str, Any]:
        """Reads metadata and metrics from a GEFF file."""
        metadata, _ = self.read_metadata_and_data(path)
        return metadata

    def get_specification(self) -> Dict[str, Any]:
        """Defines the static data types for the output metadata."""
        return {
            'geff_version': str,
            'geff_num_nodes': int,
            'geff_num_edges': int,
            'geff_num_lineages': int,
            'geff_num_divisions': int,
            'geff_num_terminations': int,
        }

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        """
        Defines patterns for dynamically generated GEFF metadata columns.
        Note: Bio-IO patterns like 'pixel_size' and '[LETTER]_size' are omitted
        as that information is not available from the GEFF spec alone.
        """
        return [
            (r'^geff_axes_\d+_name$', str),
            (r'^geff_axes_\d+_type$', str),
            (r'^geff_axes_\d+_unit$', str),
        ]