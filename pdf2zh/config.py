import json
from pathlib import Path
from threading import RLock  # Change to RLock
import os
import copy


class ConfigManager:
    _instance = None
    _lock = RLock()  # Replace Lock with RLock, allowing recursive lock acquisition in the same thread

    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        # Check if instance exists first, if not, lock and initialize
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Prevent repeated initialization
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self._config_path = Path.home() / ".config" / "PDFMathTranslate" / "config.json"
        self._config_data = {}

        # No need to lock here again, as the outer layer might have already locked (get_instance), RLock is fine
        self._ensure_config_exists()

    def _ensure_config_exists(self, isInit=True):
        """Ensure config file exists, create default if not"""
        # No need to lock here explicitly again, reason as above, _load_config() is called in the method body,
        # and _load_config() will lock internally. Since RLock is reentrant, it will not block.
        if not self._config_path.exists():
            if isInit:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                self._config_data = {}  # Default config content
                self._save_config()
            else:
                raise ValueError(f"config file {self._config_path} not found!")
        else:
            self._load_config()

    def _load_config(self):
        """Load config from config.json"""
        with self._lock:  # Lock to ensure thread safety
            with self._config_path.open("r", encoding="utf-8") as f:
                self._config_data = json.load(f)

    def _save_config(self):
        """Save config to config.json"""
        with self._lock:  # Lock to ensure thread safety
            # Remove circular references and write
            cleaned_data = self._remove_circular_references(self._config_data)
            with self._config_path.open("w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    def _remove_circular_references(self, obj, seen=None):
        """Recursively remove circular references"""
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return None  # Encountered processed object, treat as circular reference
        seen.add(obj_id)

        if isinstance(obj, dict):
            return {
                k: self._remove_circular_references(v, seen) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._remove_circular_references(i, seen) for i in obj]
        return obj

    @classmethod
    def custome_config(cls, file_path):
        """Load config file using custom path"""
        custom_path = Path(file_path)
        if not custom_path.exists():
            raise ValueError(f"Config file {custom_path} not found!")
        # Lock
        with cls._lock:
            instance = cls()
            instance._config_path = custom_path
            # Pass isInit=False here, error if not exists; _load_config() normally if exists
            instance._ensure_config_exists(isInit=False)
            cls._instance = instance

    @classmethod
    def get(cls, key, default=None):
        """Get config value"""
        instance = cls.get_instance()
        # When reading, locking or not is fine. But for consistency, we lock before and after modifying config.
        # get will lock if it eventually needs to save -> _save_config()
        if key in instance._config_data:
            return instance._config_data[key]

        # If key exists in environment variables, use it and write back to config
        if key in os.environ:
            value = os.environ[key]
            instance._config_data[key] = value
            instance._save_config()
            return value

        # If default is not None, set and save
        if default is not None:
            instance._config_data[key] = default
            instance._save_config()
            return default

        # Raise exception if not found
        # raise KeyError(f"{key} is not found in config file or environment variables.")
        return default

    @classmethod
    def set(cls, key, value):
        """Set config value and save"""
        instance = cls.get_instance()
        with instance._lock:
            instance._config_data[key] = value
            instance._save_config()

    @classmethod
    def get_translator_by_name(cls, name):
        """Get translator config by name"""
        instance = cls.get_instance()
        translators = instance._config_data.get("translators", [])
        for translator in translators:
            if translator.get("name") == name:
                return translator["envs"]
        return None

    @classmethod
    def set_translator_by_name(cls, name, new_translator_envs):
        """Set or update translator config by name"""
        instance = cls.get_instance()
        with instance._lock:
            translators = instance._config_data.get("translators", [])
            for translator in translators:
                if translator.get("name") == name:
                    translator["envs"] = copy.deepcopy(new_translator_envs)
                    instance._save_config()
                    return
            translators.append(
                {"name": name, "envs": copy.deepcopy(new_translator_envs)}
            )
            instance._config_data["translators"] = translators
            instance._save_config()

    @classmethod
    def get_env_by_translatername(cls, translater_name, name, default=None):
        """Get translator config by name"""
        instance = cls.get_instance()
        translators = instance._config_data.get("translators", [])
        for translator in translators:
            if translator.get("name") == translater_name.name:
                if translator["envs"][name]:
                    return translator["envs"][name]
                else:
                    with instance._lock:
                        translator["envs"][name] = default
                        instance._save_config()
                        return default

        with instance._lock:
            translators = instance._config_data.get("translators", [])
            for translator in translators:
                if translator.get("name") == translater_name.name:
                    translator["envs"][name] = default
                    instance._save_config()
                    return default
            translators.append(
                {
                    "name": translater_name.name,
                    "envs": copy.deepcopy(translater_name.envs),
                }
            )
            instance._config_data["translators"] = translators
            instance._save_config()
            return default

    @classmethod
    def delete(cls, key):
        """Delete config value and save"""
        instance = cls.get_instance()
        with instance._lock:
            if key in instance._config_data:
                del instance._config_data[key]
                instance._save_config()

    @classmethod
    def clear(cls):
        """Delete config value and save"""
        instance = cls.get_instance()
        with instance._lock:
            instance._config_data = {}
            instance._save_config()

    @classmethod
    def all(cls):
        """Return all config items"""
        instance = cls.get_instance()
        # Read only here, generally no need to lock. But can lock for safety.
        return instance._config_data

    @classmethod
    def remove(cls):
        instance = cls.get_instance()
        with instance._lock:
            os.remove(instance._config_path)
