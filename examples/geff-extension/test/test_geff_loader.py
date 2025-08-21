from pathlib import Path

import geff
import networkx as nx
import numpy as np
import pytest
from geff.metadata_schema import Axis

from traxel_patrol.loaders.geff_loader import GeffLoader


@pytest.fixture
def sample_geff_zarr(tmp_path: Path) -> Path:
    """Creates a temporary GEFF Zarr store for testing using the geff API."""
    geff_path = tmp_path / "test.zarr"

    # 1) Graph with node/edge attributes
    G = nx.DiGraph()
    G.add_nodes_from([
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

    # 2) GEFF metadata
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
        extra={"name": "confidence", "type": "measure"},
    )

    # 3) Write to zarr
    geff.write_nx(G, geff_path, metadata=metadata)
    return geff_path


class TestGeffLoader:
    def test_loader_instantiation(self):
        """Loader can be constructed and exposes a `load` call."""
        loader = GeffLoader()
        assert callable(getattr(loader, "load", None))

    def test_global_graph_metrics(self, sample_geff_zarr):
        """Graph-wide summary metrics are exposed in Artifact.meta."""
        loader = GeffLoader()
        art = loader.load(str(sample_geff_zarr))
        meta = art.meta

        assert meta["geff_num_nodes"] == 6
        assert meta["geff_num_edges"] == 4
        assert meta["geff_num_lineages"] == 2
        assert meta["geff_num_divisions"] == 1
        assert meta["geff_num_terminations"] == 3  # nodes 2, 4, 6 terminate

    def test_dynamic_attribute_summaries(self, sample_geff_zarr):
        """Custom numerical attribute summaries are present."""
        loader = GeffLoader()
        art = loader.load(str(sample_geff_zarr))
        meta = art.meta

        # Node attribute 'confidence'
        assert "geff_node_attr_confidence" in meta
        expected_conf_mean = np.mean([0.9, 0.95, 0.92, 0.8, 0.99, 0.98])
        assert meta["geff_node_attr_confidence"] == pytest.approx(expected_conf_mean)

        # Edge attribute 'weight'
        assert "geff_edge_attr_weight" in meta
        expected_weight_mean = np.mean([1.5, 1.8, 1.2, 2.0])
        assert meta["geff_edge_attr_weight"] == pytest.approx(expected_weight_mean)

    def test_timesliced_metrics(self, sample_geff_zarr):
        """Time-sliced metrics per T are exposed."""
        loader = GeffLoader()
        art = loader.load(str(sample_geff_zarr))
        meta = art.meta

        print(meta)

        # t = 0
        assert meta["geff_num_nodes_T0"] == 2
        assert meta["geff_num_divisions_T0"] == 1  # node 1 divides

        # t = 1
        assert meta["geff_num_nodes_T1"] == 3
        assert meta["geff_num_terminations_T1"] == 2  # nodes 2 and 6 terminate

        # time-sliced attribute summary
        expected_t1_conf_mean = np.mean([0.95, 0.92, 0.98])
        assert meta["geff_node_attr_confidence_T1"] == pytest.approx(expected_t1_conf_mean)

    def test_failure_on_invalid_path(self, tmp_path):
        """Invalid path raises an IOError via loader.load()."""
        loader = GeffLoader()
        invalid_path = tmp_path / "not_a_zarr"
        invalid_path.mkdir()
        with pytest.raises(IOError):
            loader.load(str(invalid_path))
