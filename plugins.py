"""Plugin system and user action scripting.

Plugins let users add actions without forking the application.  A plugin is a
single ``.py`` file in ``plugins/`` exposing a ``register(api)`` function::

    # plugins/my_plugin.py
    PLUGIN_NAME = "My Plugin"
    PLUGIN_VERSION = "1.0"

    def register(api):
        api.add_action("say_hello", "Say Hello", "Custom",
                       lambda ctx, event: print("hello"))

Security note
-------------
Plugins are ordinary Python modules executed in-process, so a plugin can do
anything the application can.  There is no sandbox and this file does not
pretend to provide one — the trust model is exactly that of a shell rc file.
The loader therefore never fetches code from the network and only ever
imports from the local plugin directory, so installing a plugin is always a
deliberate act of copying a file in.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Optional

from actions import ActionContext, ActionRegistry, ActionSpec
from config import PLUGIN_DIR
from gesture_engine import GestureEvent
from logger import get_logger

log = get_logger(__name__)


@dataclass
class PluginInfo:
    """Metadata about a loaded (or failed) plugin."""

    name: str
    path: Path
    version: str = "1.0"
    author: str = ""
    description: str = ""
    loaded: bool = False
    error: str = ""
    actions: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """One-word status for the plugins list."""
        return "Loaded" if self.loaded else "Failed"


class PluginAPI:
    """The surface a plugin is given at registration time.

    Deliberately narrow.  Plugins get to add actions and read state; they do
    not get the raw registry or config objects, so a badly written plugin
    cannot corrupt the application's own bindings.
    """

    def __init__(self, registry: ActionRegistry, plugin: PluginInfo) -> None:
        self._registry = registry
        self._plugin = plugin

    def add_action(self, action_id: str, label: str, category: str,
                   handler: Callable[[ActionContext, GestureEvent], bool],
                   description: str = "", sound: str = "") -> bool:
        """Register a new bindable action.

        The id is namespaced with the plugin's module name so two plugins
        cannot collide, and so a user can see where an action came from.
        """
        namespaced = f"{self._plugin.path.stem}.{action_id}"
        spec = ActionSpec(
            action_id=namespaced,
            label=label,
            category=category or "Plugins",
            handler=handler,
            description=description,
            sound=sound,
        )
        if self._registry.register(spec):
            self._plugin.actions.append(namespaced)
            log.info("plugin %r registered action %r", self._plugin.name, namespaced)
            return True
        return False

    def add_shell_action(self, action_id: str, label: str,
                         command: List[str], description: str = "") -> bool:
        """Convenience: register an action that runs a shell command."""

        def handler(ctx: ActionContext, event: GestureEvent) -> bool:
            return ctx.platform.system.run_command(command)

        return self.add_action(action_id, label, "Plugins", handler, description)

    def add_url_action(self, action_id: str, label: str, url: str,
                       description: str = "") -> bool:
        """Convenience: register an action that opens a URL."""

        def handler(ctx: ActionContext, event: GestureEvent) -> bool:
            return ctx.platform.system.open_url(url)

        return self.add_action(action_id, label, "Plugins", handler, description)

    def log_info(self, message: str) -> None:
        """Write to the application log, tagged with the plugin name."""
        log.info("[%s] %s", self._plugin.name, message)


class PluginManager:
    """Discovers, loads and reloads plugins from the plugin directory."""

    def __init__(self, registry: ActionRegistry,
                 directory: Path = PLUGIN_DIR) -> None:
        self.registry = registry
        self.directory = directory
        self.plugins: Dict[str, PluginInfo] = {}
        self._modules: Dict[str, ModuleType] = {}

    def discover(self) -> List[Path]:
        """Find candidate plugin files."""
        if not self.directory.exists():
            return []
        return sorted(
            path for path in self.directory.glob("*.py")
            if not path.name.startswith("_")
        )

    def load_all(self) -> int:
        """Load every discovered plugin.  Returns how many succeeded."""
        loaded = 0
        for path in self.discover():
            if self.load(path):
                loaded += 1
        if loaded:
            log.info("loaded %d plugin(s)", loaded)
        return loaded

    def load(self, path: Path) -> bool:
        """Load one plugin file.

        Every failure mode — syntax error, missing ``register``, exception
        during registration — is caught and recorded against the plugin
        rather than propagated.  One broken plugin must not prevent the
        application from starting.
        """
        info = PluginInfo(name=path.stem, path=path)
        self.plugins[path.stem] = info

        try:
            spec = importlib.util.spec_from_file_location(
                f"gmp_plugin_{path.stem}", path)
            if spec is None or spec.loader is None:
                info.error = "could not create module spec"
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            info.name = getattr(module, "PLUGIN_NAME", path.stem)
            info.version = str(getattr(module, "PLUGIN_VERSION", "1.0"))
            info.author = str(getattr(module, "PLUGIN_AUTHOR", ""))
            info.description = str(getattr(module, "PLUGIN_DESCRIPTION", ""))

            register = getattr(module, "register", None)
            if not callable(register):
                info.error = "no register(api) function"
                log.warning("plugin %s has no register() function", path.name)
                return False

            register(PluginAPI(self.registry, info))
            info.loaded = True
            self._modules[path.stem] = module
            return True

        except Exception as exc:
            info.error = f"{type(exc).__name__}: {exc}"
            log.error("plugin %s failed to load: %s", path.name, exc)
            log.debug("%s", traceback.format_exc())
            return False

    def unload(self, name: str) -> bool:
        """Unregister a plugin's actions and drop its module."""
        info = self.plugins.get(name)
        if info is None:
            return False

        for action_id in info.actions:
            self.registry.unregister(action_id)
        info.actions.clear()
        info.loaded = False

        module_name = f"gmp_plugin_{name}"
        self._modules.pop(name, None)
        sys.modules.pop(module_name, None)
        log.info("unloaded plugin %r", name)
        return True

    def reload(self, name: str) -> bool:
        """Unload and re-load a plugin, picking up source edits."""
        info = self.plugins.get(name)
        if info is None:
            return False
        self.unload(name)
        return self.load(info.path)

    def reload_all(self) -> int:
        """Reload every known plugin."""
        for name in list(self.plugins):
            self.unload(name)
        return self.load_all()

    @property
    def loaded_plugins(self) -> List[PluginInfo]:
        """Successfully loaded plugins."""
        return [p for p in self.plugins.values() if p.loaded]

    @property
    def failed_plugins(self) -> List[PluginInfo]:
        """Plugins that failed, with their errors."""
        return [p for p in self.plugins.values() if not p.loaded]

    def write_example(self) -> Optional[Path]:
        """Write a documented example plugin if none exists."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "example_plugin.py"
        if path.exists():
            return None

        try:
            path.write_text(_EXAMPLE_PLUGIN, encoding="utf-8")
            log.info("wrote example plugin to %s", path)
            return path
        except OSError as exc:
            log.warning("could not write example plugin: %s", exc)
            return None


_EXAMPLE_PLUGIN = '''"""Example plugin for AI Gesture Mouse Pro.

Copy this file, rename it, and edit ``register`` to add your own actions.
Every action you register appears in Settings -> Gesture Bindings and can be
bound to any gesture, including ones you record yourself.
"""

PLUGIN_NAME = "Example Plugin"
PLUGIN_VERSION = "1.0"
PLUGIN_AUTHOR = "You"
PLUGIN_DESCRIPTION = "Demonstrates the three ways to register an action."


def register(api):
    """Called once at startup with a PluginAPI instance."""

    # 1. A full custom handler.  It receives the action context (cursor,
    #    platform bridge, notifier, ...) and the gesture event that fired it.
    def take_notes(ctx, event):
        ctx.notify("Plugin", f"Fired by {event.name} at {event.confidence:.0%}")
        return True

    api.add_action(
        "take_notes", "Show a Notification", "Plugins",
        take_notes, description="Displays a toast notification.",
    )

    # 2. Run a shell command.
    api.add_shell_action(
        "open_downloads", "Open Downloads Folder",
        ["open", "-a", "Finder", "~/Downloads"],   # adjust for your OS
        description="Opens the Downloads folder.",
    )

    # 3. Open a URL.
    api.add_url_action(
        "open_docs", "Open Documentation",
        "https://github.com/", description="Opens a web page.",
    )

    api.log_info("example plugin registered 3 actions")
'''
