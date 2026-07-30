import json
import threading
import typing
from pathlib import Path
from camera import DEFAULT_CONFIG
from blob_detector import DEF

CONFIG_FILE = "config.json"
ConfigValue = float | int

class Config:
    def __init__(self, config_file: str = CONFIG_FILE) -> None:
        self._config_path = Path(config_file)
        self._lock = threading.Lock()
        with self._lock:
            self._values: typing.Dict[str, ConfigValue] = self._load_file()
        self._defaults: typing.Dict[str, ConfigValue] = {**}
        self._callbacks: typing.List[typing.Callable[[str, ConfigValue], typing.Any]] = []

    def _load_file(self) -> typing.Dict[str, ConfigValue]:
        if not self._config_path.exists():
            return {}

        try:
            with self._config_path.open("r", encoding="utf-8") as config_file:
                return json.load(config_file)
        except (OSError, json.JSONDecodeError):
            return {}


    def set_default(self, name: str, default: ConfigValue):
        self._defaults[name] = default

    def get(self, name: str) -> ConfigValue:
        with self._lock:
            return self._values[name]

    def get_default(self, name: str, default: int | float) -> ConfigValue:
        with self._lock:
            return self._values.get(name, default)

    def set(self, name: str, value: ConfigValue) -> None:
        with self._lock:
            self._values[name] = value
        for cb in self._callbacks:
            cb(name, value)

    def save_file(self) -> None:
        with self._config_path.open("w", encoding="utf-8") as config_file:
            json.dump(self._values, config_file)

    def as_dict(self) -> typing.Dict[str, ConfigValue]:
        return self._values.copy()

    def add_callback(self, cb: typing.Callable[[str, ConfigValue], typing.Any]):
        self._callbacks.append(cb)

    def remove_callback(self, cb: typing.Callable):
        if cb in self._callbacks:
            self._callbacks.remove(cb)