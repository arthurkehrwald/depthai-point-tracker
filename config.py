import threading
import tomllib
import typing
from pathlib import Path

import tomli_w

CONFIG_FILE = "config.toml"
ConfigValue = float | int


class Config:
    def __init__(self, config_file: str = CONFIG_FILE, defaults: typing.Dict[str, ConfigValue] | None = None) -> None:
        self._config_path = Path(config_file)
        self._lock = threading.Lock()
        with self._lock:
            self._values: typing.Dict[str, ConfigValue] = self._load_file()
        self._defaults = defaults if defaults is not None else {}
        self._callbacks: typing.List[typing.Callable[[typing.List[str]], typing.Any]] = []
        self.changed_since_last_callback: typing.List[str] = []

    def _load_file(self) -> typing.Dict[str, ConfigValue]:
        if not self._config_path.exists():
            return {}

        try:
            with self._config_path.open("rb") as config_file:
                return tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError):
            return {}

    def get(self, name: str) -> ConfigValue:
        with self._lock:
            return self._values.get(name, self._defaults[name])

    def set(self, name: str, value: ConfigValue) -> None:
        with self._lock:
            self._values[name] = value
            if name not in self.changed_since_last_callback:
                self.changed_since_last_callback.append(name)

    def save_file(self) -> None:
        with self._config_path.open("wb") as config_file:
            with self._lock:
                # Keys are StrEnum members; TOML needs plain strings.
                tomli_w.dump({str(name): value for name, value in self._values.items()}, config_file)

    def add_callback(self, cb: typing.Callable[[typing.List[str]], typing.Any]):
        with self._lock:
            self._callbacks.append(cb)

    def remove_callback(self, cb: typing.Callable):
        with self._lock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)

    def do_callbacks(self):
        with self._lock:
            if not self.changed_since_last_callback:
                return
            changed_copy = self.changed_since_last_callback.copy()
            self.changed_since_last_callback.clear()
        for cb in self._callbacks:
            cb(changed_copy)
