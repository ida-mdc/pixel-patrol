from pathlib import Path

import geff
import networkx as nx
import numpy as np
import pytest
from geff.metadata_schema import Axis

from pixel_patrol.core.loaders.geff_loader import GeffLoader


@pytest.fixture
def sample_geff_zarr(tmp_path: Path) -> Path:
    """Creates a temporary GEFF Zarr store for testing using the geff API."""
    geff_path = tmp_path / "test.zarr"

    # 1. Define the complete NetworkX graph with all node and edge attributes
    G = nx.DiGraph()
    G.add_nodes_from([
        # Node ID, {attributes dictionary}
        (1, {"POSITION_T": 0, "POSITION_Z": 10, "POSITION_Y": 10, "POSITION_X": 10, "confidence": 0.9}),
        (2, {"POSITION_T": 1, "POSITION_Z": 11, "POSITION_Y": 11, "POSITION_X": 11, "confidence": 0.95}),
        (3, {"POSITION_T": 1, "POSITION_Z": 12, "POSITION_Y": 12, "POSITION_X": 12, "confidence": 0.92}),
        (4, {"POSITION_T": 2, "POSITION_Z": 13, "POSITION_Y": 13, "POSITION_X": 13, "confidence": 0.8}),
        (5, {"POSITION_T": 0, "POSITION_Z": 20, "POSITION_Y": 20, "POSITION_X": 20, "confidence": 0.99}),
        (6, {"POSITION_T": 1, "POSITION_Z": 21, "POSITION_Y": 21, "POSITION_X": 21, "confidence": 0.98}),
    ])
    G.add_edges_from([
        (1, 2, {"weight": 1.5}),
        (1, 3, {"weight": 1.8}),
        (3, 4, {"weight": 1.2}),
        (5, 6, {"weight": 2.0}),
    ])

    # 2. Define the GEFF metadata that describes the graph's axes
    # The geff library automatically prepends the 'id' axis.
    metadata = geff.GeffMetadata(
        geff_version="0.1.0",
        axes=[
            Axis(name="POSITION_X", type="space"),
            Axis(name="POSITION_Y", type="space"),
            Axis(name="POSITION_Z", type="space"),
            Axis(name="POSITION_T", type="time"),
        ],
        directed=True,
        track_node_props={"lineage": "TRACK_ID"},
        extra={"name": "confidence", "type": "measure"}
    )

    # 3. Write the entire graph and metadata to the Zarr store in one call
    geff.write_nx(G, geff_path, metadata=metadata)

    return geff_path


# The TestGeffLoader class remains the same as it was already correct.
class TestGeffLoader:
    # ... (all previous test methods are unchanged) ...
    def test_loader_instantiation(self):
        """Tests that the loader can be created and has correct properties."""
        loader = GeffLoader()
        assert loader.reads_only_metadata is True

    def test_global_graph_metrics(self, sample_geff_zarr):
        """Tests the calculation of graph-wide summary metrics."""
        loader = GeffLoader()
        metadata, _ = loader.read_metadata_and_data(sample_geff_zarr)

        assert metadata["geff_metric_num_nodes"] == 6
        assert metadata["geff_metric_num_edges"] == 4
        assert metadata["geff_metric_num_lineages"] == 2
        assert metadata["geff_metric_num_divisions"] == 1
        assert metadata["geff_metric_num_terminations"] == 3  # Nodes 2, 4, 6 terminate

    def test_dynamic_attribute_summaries(self, sample_geff_zarr):
        """Tests the summarization of custom numerical attributes."""
        loader = GeffLoader()
        metadata, _ = loader.read_metadata_and_data(sample_geff_zarr)

        # Test node attribute 'confidence'
        assert "geff_node_attr_confidence_mean" in metadata
        expected_confidence_mean = np.mean([0.9, 0.95, 0.92, 0.8, 0.99, 0.98])
        assert metadata["geff_node_attr_confidence_mean"] == pytest.approx(expected_confidence_mean)
        assert metadata["geff_node_attr_confidence_min"] == 0.8

        # Test edge attribute 'weight'
        assert "geff_edge_attr_weight_mean" in metadata
        expected_weight_mean = np.mean([1.5, 1.8, 1.2, 2.0])
        assert metadata["geff_edge_attr_weight_mean"] == pytest.approx(expected_weight_mean)
        assert metadata["geff_edge_attr_weight_max"] == 2.0

    def test_timesliced_metrics(self, sample_geff_zarr):
        """Tests the calculation of metrics per timepoint."""
        loader = GeffLoader()
        metadata, _ = loader.read_metadata_and_data(sample_geff_zarr)

        # Test timepoint t=0
        assert metadata["geff_t_0_num_nodes"] == 2
        assert metadata["geff_t_0_divisions"] == 1  # Node 1 divides

        # Test timepoint t=1
        assert metadata["geff_t_1_num_nodes"] == 3
        assert metadata["geff_t_1_terminations"] == 2  # Nodes 2 and 6 terminate here

        # Test time-sliced attribute summary
        assert "geff_t_1_attr_confidence_mean" in metadata
        expected_t1_confidence_mean = np.mean([0.95, 0.92, 0.98])
        assert metadata["geff_t_1_attr_confidence_mean"] == pytest.approx(expected_t1_confidence_mean)

    def test_failure_on_invalid_path(self, tmp_path):
        """Tests that the loader raises an IOError for a non-GEFF path."""
        loader = GeffLoader()
        invalid_path = tmp_path / "not_a_zarr"
        invalid_path.mkdir()

        with pytest.raises(IOError):
            loader.read_metadata_and_data(invalid_path)