"""Every registered loader and processor must describe its output columns.

Lives in the umbrella package because it depends on all plugin packages, so the
check covers every installed loader and processor without enumerating them.
Descriptions feed the schema catalog export and the parquet field metadata, so a
missing one is a documentation gap worth failing on.
"""

import importlib.metadata

import pytest

from pixel_patrol_base.core import schema_catalog


def _discover(group):
    out = []
    for ep in importlib.metadata.entry_points(group=group):
        try:
            for cls in ep.load()():
                out.append(cls)
        except Exception:
            continue
    return out


_LOADERS = _discover("pixel_patrol.loader_plugins")
_PROCESSORS = _discover("pixel_patrol.processor_plugins")
_COMPONENTS = [(getattr(c, "NAME", c.__name__), c) for c in _LOADERS + _PROCESSORS]


def test_some_components_registered():
    assert _COMPONENTS, "No loaders/processors registered in this environment."


@pytest.mark.parametrize("component", [c for _, c in _COMPONENTS],
                         ids=[name for name, _ in _COMPONENTS])
def test_every_output_column_has_a_description(component):
    schema = getattr(component, "OUTPUT_SCHEMA", {}) or {}
    missing = [
        col for col in schema
        if not schema_catalog._resolve_column_description(component, col)
    ]
    assert not missing, (
        f"{getattr(component, 'NAME', component)} has OUTPUT_SCHEMA columns without "
        f"a description: {missing}"
    )


@pytest.mark.parametrize("component", [c for _, c in _COMPONENTS],
                         ids=[name for name, _ in _COMPONENTS])
def test_component_has_self_description(component):
    assert getattr(component, "DESCRIPTION", "").strip(), (
        f"{getattr(component, 'NAME', component)} is missing a DESCRIPTION."
    )
