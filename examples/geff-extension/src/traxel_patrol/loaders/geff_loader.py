import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import geff
import networkx as nx
import numpy as np

from pixel_patrol_base.core.artifact import Artifact

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
        summary_stats[f"geff_{prefix}"] = float(np.mean(values_np))

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
        summary_stats[f"geff_{prefix}"] = float(np.mean(values_np))

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
            timesliced_metrics[key + prefix] = value

    return timesliced_metrics


def _extract_axis_information(nx_graph: nx.DiGraph, geff_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts axis dimension order, names, and sizes from the GEFF spec and graph nodes.
    Produces fields similar to image-based loaders: dim_order and per-axis sizes.
    """
    axis_metadata: Dict[str, Any] = {}
    if 'axes' not in geff_spec or not isinstance(geff_spec['axes'], list):
        return axis_metadata

    axis_names_map: Dict[str, str] = {}
    dim_order_list: List[str] = []

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


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key and not k.startswith(parent_key) else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(_flatten_dict(item, f"{new_key}_{i}", sep=sep))
                else:
                    items[f"{new_key}_{i}"] = item
        else:
            items[new_key] = v
    return items


class GeffLoader:
    """
    Artifact-first GEFF loader.

    - Single entrypoint: `load(source) -> Artifact`
    - Returns a graph-kind Artifact (data = networkx graph), with axes inferred from GEFF spec (if present).
    - Adds rich, flattened metadata so it flows into your features table.
    """

    NAME = "geff"

    SUPPORTED_EXTENSIONS: Set[str, Any] = ["zarr"]

    OUTPUT_SCHEMA = {
        'geff_version': str,
        'geff_num_nodes': int,
        'geff_num_edges': int,
        'geff_num_lineages': int,
        'geff_num_divisions': int,
        'geff_num_terminations': int,
        'dim_order': str,
        'axis_names': dict,
    }
    OUTPUT_SCHEMA_PATTERNS = [
        (r'^[a-zA-Z]_size$', int),
        (r'^geff_(node|edge)_attr_.*$', float),
        (r'^geff_num_(nodes|divisions|terminations)_T\d+$', int),  # time-sliced counts
    ]

    def load(self, source: str) -> Artifact:
        path = Path(source)
        logger.debug(f"Attempting to read '{path}' with GeffLoader.")

        try:
            nx_graph, geff_meta_obj = geff.read_nx(path)
            geff_spec = (
                geff_meta_obj.model_dump()
                if hasattr(geff_meta_obj, 'model_dump')
                else geff_meta_obj.dict()
            )
        except Exception as e:
            raise IOError(f"Failed to read '{path}' as a GEFF file. Error: {e}")

        # Aggregate metadata
        metadata: Dict[str, Any] = {}
        metadata.update(_flatten_dict(geff_spec, parent_key='geff'))
        metadata.update(_extract_axis_information(nx_graph, geff_spec))
        metadata.update(_calculate_global_graph_metrics(nx_graph))
        metadata.update(_summarize_numerical_attributes(nx_graph.nodes(data=True), "node"))
        metadata.update(_summarize_numerical_attributes_edges(nx_graph.edges(data=True), "edge"))
        metadata.update(_calculate_timesliced_metrics(nx_graph, geff_spec))

        # Build axes/capabilities from dim_order (if provided)
        dim_order = metadata.get("dim_order", "")
        axes: Set[str] = set(dim_order) if isinstance(dim_order, str) else set()
        caps: Set[str] = set()
        if "X" in axes and "Y" in axes:
            caps.add("spatial-2d")
        if "Z" in axes:
            caps.add("spatial-3d")
        if "T" in axes:
            caps.add("temporal")

        logger.info(f"Successfully parsed GEFF metadata from '{path}'.")
        return Artifact(
            data=nx_graph,
            axes=axes,
            kind="graph",
            meta=metadata,
            capabilities=caps,
        )
