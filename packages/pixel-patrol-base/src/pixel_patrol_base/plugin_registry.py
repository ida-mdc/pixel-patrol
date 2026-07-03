from __future__ import annotations

import importlib
import importlib.metadata
import logging
from typing import TYPE_CHECKING, List, Type, Union

if TYPE_CHECKING:
    from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor, PixelPatrolSource
    PixelPluginClass = Union[Type[PixelPatrolLoader], Type[PixelPatrolProcessor], Type[PixelPatrolSource]]

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_ID = "local_filesystem"

def discover_loader(loader_id: str) -> PixelPatrolLoader:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.loader_plugins")
    logger.debug(f'Discovered loader plugins: {", ".join([plugin.NAME for plugin in plugins])}')
    for loader_plugin in plugins:
        if loader_plugin.NAME == loader_id:
            return loader_plugin()
    raise RuntimeError(f'Could not find loader plugin "{loader_id}" in discovered loader plugins: {[plugin.NAME for plugin in plugins]}')

def discover_sources() -> List[PixelPatrolSource]:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.source_plugins")
    initialized_plugins = [plugin() for plugin in plugins]
    # Base's built-in sources ship in-package; guarantee they are always
    # candidates even if entry-point enumeration was disrupted (e.g. by a
    # re-imported package in a test) and missed them.
    names = {s.NAME for s in initialized_plugins}
    for builtin_cls in register_source_plugins():
        if builtin_cls.NAME not in names:
            initialized_plugins.append(builtin_cls())
            names.add(builtin_cls.NAME)
    logger.debug(f'Discovered source plugins: {", ".join([s.NAME for s in initialized_plugins])}')
    return initialized_plugins


def select_source(bases, source_id: str | None = None) -> PixelPatrolSource:
    """Return the source plugin to use for the given bases.

    With source_id set, return that named source. Otherwise auto-route: pick the
    single source whose can_handle accepts every base. Raises if the choice is
    absent or ambiguous so a run never silently uses the wrong source.
    """
    sources = discover_sources()

    if source_id is not None:
        for source in sources:
            if source.NAME == source_id:
                return source
        raise RuntimeError(f'Could not find source plugin "{source_id}" in: {[s.NAME for s in sources]}')

    matches = [s for s in sources if all(s.can_handle(str(b)) for b in bases)]
    if not matches:
        raise RuntimeError(f"No source plugin can handle bases: {list(bases)}")
    if len(matches) == 1:
        return matches[0]

    # The local source is the universal fallback; a specialised source that also
    # matches (e.g. ManifestSource claiming a .csv base) is more specific and wins.
    specific = [s for s in matches if s.NAME != DEFAULT_SOURCE_ID]
    if len(specific) == 1:
        return specific[0]
    raise RuntimeError(f"Ambiguous source selection for {list(bases)}: {[s.NAME for s in matches]}")


def register_source_plugins():
    from pixel_patrol_base.plugins.sources.local_filesystem_source import LocalFilesystemSource
    from pixel_patrol_base.plugins.sources.manifest_source import ManifestSource
    return [
        LocalFilesystemSource,
        ManifestSource,
    ]


def discover_processor_plugins() -> List[PixelPatrolProcessor]:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.processor_plugins")
    initialized_plugins = [plugin() for plugin in plugins]
    logger.debug(f'Discovered processor plugins: {", ".join([plugin.NAME for plugin in initialized_plugins])}')
    return initialized_plugins


def discover_plugins_from_entrypoints(plugins_id) -> List[PixelPluginClass]:
    res: List[PixelPluginClass] = []
    entry_points = importlib.metadata.entry_points(group=plugins_id)
    for ep in entry_points:
        try:
            registration_func = ep.load()
            components = registration_func()
            res.extend(components)
        except Exception as e:
            logger.error(f"Could not load plugin '{ep.name}': {e}")
    return res

def register_processor_plugins():
    from pixel_patrol_base.plugins.processors.raster_processor import BasicMetricsProcessor, HistogramProcessor
    from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor
    return [
        BasicMetricsProcessor,
        HistogramProcessor,
        ThumbnailProcessor,
    ]

