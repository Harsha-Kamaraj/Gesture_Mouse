"""Profile management — load, save, import, export and switch configurations.

A *profile* is a complete :class:`~config.AppConfig` stored as one JSON file
in ``profiles/``.  Filenames are derived from the profile name via slugging,
so the on-disk layout stays human-readable and a user can hand-edit or
version-control their settings.

Switching profiles is a first-class operation: subscribers are notified so the
detector, cursor controller and UI all re-read their settings without the
application restarting.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from config import (
    APP_STATE_FILE, AppConfig, PROFILES_DIR, builtin_profiles,
)
from logger import get_logger

log = get_logger(__name__)

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Convert a profile name into a safe filename stem."""
    slug = _SLUG_PATTERN.sub("-", name.strip().lower()).strip("-")
    return slug or "profile"


class ProfileError(Exception):
    """Raised for recoverable profile operations (duplicate name, bad file)."""


class ProfileManager:
    """Owns the set of profiles and the currently active configuration."""

    def __init__(self, directory: Path = PROFILES_DIR) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._config: AppConfig = AppConfig()
        self._lock = threading.RLock()
        self._subscribers: List[Callable[[AppConfig], None]] = []
        self.seed_builtins()

    # -- discovery -------------------------------------------------------- #

    def seed_builtins(self) -> int:
        """Write the shipped presets on first run.  Returns how many were created."""
        created = 0
        for profile in builtin_profiles():
            path = self.path_for(profile.profile_name)
            if not path.exists():
                profile.save(path)
                created += 1
        if created:
            log.info("seeded %d built-in profiles", created)
        return created

    def path_for(self, name: str) -> Path:
        """Filesystem path for a profile name."""
        return self.directory / f"{slugify(name)}.json"

    def list_profiles(self) -> List[str]:
        """Names of every profile on disk, alphabetically."""
        names: List[str] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                names.append(str(data.get("profile_name", path.stem)))
            except (OSError, json.JSONDecodeError):
                log.warning("skipping unreadable profile %s", path.name)
        return sorted(names)

    def describe_profiles(self) -> Dict[str, str]:
        """``name -> description`` for the profile picker."""
        out: Dict[str, str] = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out[str(data.get("profile_name", path.stem))] = \
                    str(data.get("description", ""))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    # -- active config ---------------------------------------------------- #

    @property
    def config(self) -> AppConfig:
        """The currently active configuration."""
        with self._lock:
            return self._config

    @property
    def active_name(self) -> str:
        """Name of the active profile."""
        return self._config.profile_name

    def subscribe(self, callback: Callable[[AppConfig], None]) -> None:
        """Register a callback fired whenever the active config changes."""
        self._subscribers.append(callback)

    def _notify(self) -> None:
        """Fire every subscriber with the current config."""
        for callback in list(self._subscribers):
            try:
                callback(self._config)
            except Exception as exc:
                log.warning("profile subscriber failed: %s", exc)

    # -- operations ------------------------------------------------------- #

    def load(self, name: str) -> AppConfig:
        """Activate a profile by name."""
        path = self.path_for(name)
        if not path.exists():
            raise ProfileError(f"Profile {name!r} does not exist")

        with self._lock:
            self._config = AppConfig.load(path)
            # Guard against a hand-edited file whose name no longer matches.
            self._config.profile_name = name
        log.info("activated profile %r", name)
        self._notify()
        self.remember_last_used(name)
        return self._config

    def save(self, config: Optional[AppConfig] = None) -> Path:
        """Persist a config (defaults to the active one)."""
        with self._lock:
            target = config or self._config
            path = self.path_for(target.profile_name)
            target.save(path)
        log.info("saved profile %r", target.profile_name)
        return path

    def create(self, name: str, base: Optional[AppConfig] = None) -> AppConfig:
        """Create a new profile, optionally cloning ``base``."""
        if not name.strip():
            raise ProfileError("Profile name cannot be empty")
        path = self.path_for(name)
        if path.exists():
            raise ProfileError(f"A profile named {name!r} already exists")

        profile = (base or self._config).clone(new_name=name)
        profile.save(path)
        log.info("created profile %r", name)
        return profile

    def duplicate(self, name: str, new_name: Optional[str] = None) -> AppConfig:
        """Copy an existing profile under a new, non-colliding name."""
        source_path = self.path_for(name)
        if not source_path.exists():
            raise ProfileError(f"Profile {name!r} does not exist")

        source = AppConfig.load(source_path)
        target_name = new_name or self._unique_name(f"{name} Copy")
        return self.create(target_name, base=source)

    def _unique_name(self, candidate: str) -> str:
        """Append a counter until the name is free."""
        if not self.path_for(candidate).exists():
            return candidate
        for index in range(2, 100):
            attempt = f"{candidate} {index}"
            if not self.path_for(attempt).exists():
                return attempt
        raise ProfileError("Could not find a free profile name")

    def rename(self, old: str, new: str) -> AppConfig:
        """Rename a profile, moving its file."""
        if not new.strip():
            raise ProfileError("Profile name cannot be empty")
        source = self.path_for(old)
        target = self.path_for(new)
        if not source.exists():
            raise ProfileError(f"Profile {old!r} does not exist")
        if target.exists() and target != source:
            raise ProfileError(f"A profile named {new!r} already exists")

        config = AppConfig.load(source)
        config.profile_name = new
        config.save(target)
        if target != source:
            source.unlink()

        with self._lock:
            if self._config.profile_name == old:
                self._config = config
        log.info("renamed profile %r -> %r", old, new)
        self._notify()
        return config

    def delete(self, name: str) -> bool:
        """Delete a profile.

        The active profile cannot be deleted, and neither can the last
        remaining one — either would leave the application with no
        configuration to fall back to.
        """
        if name == self.active_name:
            raise ProfileError("Cannot delete the active profile; switch first")
        if len(self.list_profiles()) <= 1:
            raise ProfileError("Cannot delete the only remaining profile")

        path = self.path_for(name)
        if not path.exists():
            return False
        path.unlink()
        log.info("deleted profile %r", name)
        return True

    # -- import / export -------------------------------------------------- #

    def export(self, name: str, destination: Path) -> Path:
        """Copy a profile out to an arbitrary path."""
        source = self.path_for(name)
        if not source.exists():
            raise ProfileError(f"Profile {name!r} does not exist")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        log.info("exported profile %r to %s", name, destination)
        return destination

    def import_profile(self, source: Path, rename_to: Optional[str] = None) -> AppConfig:
        """Import a profile file, avoiding name collisions."""
        if not source.exists():
            raise ProfileError(f"{source} does not exist")

        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"Not a valid profile file: {exc}") from exc

        config = AppConfig.from_dict(data)
        name = rename_to or config.profile_name or source.stem
        config.profile_name = self._unique_name(name)
        config.save(self.path_for(config.profile_name))
        log.info("imported profile %r", config.profile_name)
        return config

    # -- session persistence ---------------------------------------------- #

    def remember_last_used(self, name: str) -> None:
        """Record the active profile so the next launch restores it."""
        try:
            APP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state: Dict[str, object] = {}
            if APP_STATE_FILE.exists():
                try:
                    state = json.loads(APP_STATE_FILE.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    state = {}
            state["last_profile"] = name
            APP_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except OSError as exc:
            log.debug("could not persist last profile: %s", exc)

    def last_used(self) -> Optional[str]:
        """The profile active when the app last closed, if still present."""
        try:
            if APP_STATE_FILE.exists():
                state = json.loads(APP_STATE_FILE.read_text(encoding="utf-8"))
                name = state.get("last_profile")
                if isinstance(name, str) and self.path_for(name).exists():
                    return name
        except (OSError, json.JSONDecodeError):
            pass
        return None

    def load_startup_profile(self) -> AppConfig:
        """Activate the last-used profile, or the first available one."""
        name = self.last_used()
        if name is None:
            available = self.list_profiles()
            name = "Default" if "Default" in available else \
                (available[0] if available else None)

        if name is None:
            log.warning("no profiles found; using in-memory defaults")
            with self._lock:
                self._config = AppConfig()
            return self._config
        return self.load(name)

    # -- partial updates -------------------------------------------------- #

    def update_section(self, section: str, **values: object) -> AppConfig:
        """Apply keyword updates to one config section and persist.

        Used by every settings control, so the UI never mutates the config
        object directly and every change is saved and broadcast consistently.
        """
        with self._lock:
            target = getattr(self._config, section, None)
            if target is None or not is_dataclass(target):
                raise ProfileError(f"Unknown config section {section!r}")

            valid = set(asdict(target).keys())
            for key, value in values.items():
                if key not in valid:
                    log.warning("ignoring unknown setting %s.%s", section, key)
                    continue
                setattr(target, key, value)

        self.save()
        self._notify()
        return self._config

    def reset_to_defaults(self) -> AppConfig:
        """Reset the active profile to factory settings, keeping its name."""
        with self._lock:
            name = self._config.profile_name
            description = self._config.description
            self._config = AppConfig(profile_name=name, description=description)
        self.save()
        self._notify()
        log.info("profile %r reset to defaults", self.active_name)
        return self._config
