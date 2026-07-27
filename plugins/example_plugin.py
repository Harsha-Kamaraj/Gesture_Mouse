"""Example plugin for AI Gesture Mouse Pro.

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
