import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional

import dask.array as da
import geff
import networkx as nx
import numpy as np

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
        summary_stats[f"geff_{prefix}"] = np.mean(values_np)

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
        summary_stats[f"geff_{prefix}"] = np.mean(values_np)

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
        prefix = f"_T{int(t)}"
        timesliced_metrics[f"geff_num_nodes{prefix}"] = len(nodes_at_t)

        out_degrees = [nx_graph.out_degree(n_id) for n_id, _ in nodes_at_t]
        timesliced_metrics[f"geff_num_divisions{prefix}"] = sum(1 for d in out_degrees if d > 1)
        timesliced_metrics[f"geff_num_terminations{prefix}"] = sum(1 for d in out_degrees if d == 0)

        # Dynamically summarize all numerical attributes for this time slice
        node_data_at_t = (data for _, data in nodes_at_t)
        slice_attr_stats = _summarize_numerical_attributes(zip(range(len(nodes_at_t)), node_data_at_t), "node")

        # Rename keys to be time-specific
        for key, value in slice_attr_stats.items():
            new_key = key + prefix
            timesliced_metrics[new_key] = value

    return timesliced_metrics


def _extract_axis_information(nx_graph: nx.DiGraph, geff_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts axis dimension order, names, and sizes from the GEFF spec and graph nodes.
    This mimics the output of bioio for dimensions.
    """
    axis_metadata = {}
    if 'axes' not in geff_spec or not isinstance(geff_spec['axes'], list):
        return {}

    axis_names_map = {}
    dim_order_list = []

    axis_full_names = {ax.get('name') for ax in geff_spec['axes'] if ax.get('name')}
    node_attr_values = defaultdict(list)
    for _, data in nx_graph.nodes(data=True):
        for key, value in data.items():
            if key in axis_full_names:
                node_attr_values[key].append(value)

    for axis in geff_spec['axes']:
        full_name = axis.get('name')
        if not full_name:
            continue

        single_letter = full_name[0].upper()
        if single_letter in axis_names_map:
            logger.warning(
                f"Duplicate single-letter axis '{single_letter}' from '{full_name}'. "
                f"Skipping to avoid overwriting '{axis_names_map[single_letter]}'."
            )
            continue

        dim_order_list.append(single_letter)
        axis_names_map[single_letter] = full_name

        size = 1
        if full_name in node_attr_values:
            numeric_vals = [v for v in node_attr_values[full_name] if isinstance(v, (int, float))]
            if numeric_vals:
                size = int(max(numeric_vals)) + 1

        axis_metadata[f"{single_letter}_size"] = size

    axis_metadata['dim_order'] = "".join(dim_order_list)
    axis_metadata['axis_names'] = axis_names_map

    return axis_metadata


class GeffLoader(PixelPatrolLoader):

    @staticmethod
    def id() -> str:
        return "geff"

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

        # Extract axis information (dim_order, sizes, etc.) like bioio
        metadata.update(_extract_axis_information(nx_graph, geff_spec))

        # Add computed graph metrics
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
            'dim_order': str,
            'axis_names': dict,
        }

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        """
        Defines patterns for dynamically generated GEFF metadata columns.
        """
        return [
            (r'^[a-zA-Z]_size$', int),
            (r'^geff_(node|edge)_attr_.*$', float),
        ]
