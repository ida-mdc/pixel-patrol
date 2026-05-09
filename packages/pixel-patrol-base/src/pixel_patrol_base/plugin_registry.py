from __future__ import annotations

import importlib
import importlib.metadata
import logging
from typing import TYPE_CHECKING, Type, Union, List

if TYPE_CHECKING:
    from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor, PixelPatrolWidget

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    PixelPluginClass = Union[Type[PixelPatrolLoader], Type[PixelPatrolProcessor], Type[PixelPatrolWidget]]

def discover_loader(loader_id: str) -> PixelPatrolLoader:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.loader_plugins")
    logger.debug(f'Discovered loader plugins: {", ".join([plugin.NAME for plugin in plugins])}')
    for loader_plugin in plugins:
        if loader_plugin.NAME == loader_id:
            return loader_plugin()
    raise RuntimeError(f'Could not find loader plugin "{loader_id}" in discovered loader plugins: {[plugin.NAME for plugin in plugins]}')

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
    return []

