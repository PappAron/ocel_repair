import json
import sqlite3
import logging
from pathlib import Path

from core.domain import Event, EventObjectLink, Object, ObjectObjectLink, OcelLog
from core.ports import DataLoaderPort

logger = logging.getLogger(__name__)

class JsonLoader(DataLoaderPort):
    def load(self, source_path: str | Path) -> OcelLog:
        logger.info("Opening JSON OCEL log: %s", source_path)
        with open(source_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        events = tuple(
            Event(event_id, value.get("ocel:activity", value.get("type", "")), value)
            for event_id, value in data.get("ocel:events", {}).items()
        )
        objects = tuple(
            Object(object_id, value.get("ocel:type", value.get("type", "")), value)
            for object_id, value in data.get("ocel:objects", {}).items()
        )
        links = tuple(
            EventObjectLink(event_id, object_id)
            for event_id, value in data.get("ocel:events", {}).items()
            for object_id in value.get("ocel:omap", [])
        )
        return OcelLog(events, objects, links, ())

class SqliteLoader(DataLoaderPort):
    def load(self, source_path: str | Path) -> OcelLog:
        logger.info("Opening SQLite OCEL database: %s", source_path)
        connection = sqlite3.connect(source_path)
        connection.row_factory = sqlite3.Row
        try:
            event_map = self._type_map(connection, "event_map_type")
            object_map = self._type_map(connection, "object_map_type")
            events = self._load_events(connection, event_map)
            objects = self._load_objects(connection, object_map)
            event_links = tuple(
                EventObjectLink(row["ocel_event_id"], row["ocel_object_id"], row["ocel_qualifier"])
                for row in connection.execute("SELECT * FROM event_object")
            )
            object_links = tuple(
                ObjectObjectLink(
                    row["ocel_source_id"], row["ocel_target_id"], row["ocel_qualifier"]
                )
                for row in connection.execute("SELECT * FROM object_object")
            )
            logger.info(
                "SQLite normalization complete: event_types=%d object_types=%d",
                len(event_map), len(object_map),
            )
            return OcelLog(events, objects, event_links, object_links)
        finally:
            connection.close()

    @staticmethod
    def _type_map(connection: sqlite3.Connection, table: str) -> dict[str, str]:
        return {
            row["ocel_type"]: row["ocel_type_map"]
            for row in connection.execute(f"SELECT * FROM {table}")
        }

    @classmethod
    def _load_events(
        cls, connection: sqlite3.Connection, type_map: dict[str, str]
    ) -> tuple[Event, ...]:
        rows = connection.execute("SELECT * FROM event").fetchall()
        return tuple(
            Event(row["ocel_id"], row["ocel_type"], cls._extension_attributes(
                connection, f"event_{type_map.get(row['ocel_type'], '')}", row["ocel_id"]
            ))
            for row in rows
        )

    @classmethod
    def _load_objects(
        cls, connection: sqlite3.Connection, type_map: dict[str, str]
    ) -> tuple[Object, ...]:
        rows = connection.execute("SELECT * FROM object").fetchall()
        return tuple(
            Object(row["ocel_id"], row["ocel_type"], cls._extension_attributes(
                connection, f"object_{type_map.get(row['ocel_type'], '')}", row["ocel_id"]
            ))
            for row in rows
        )

    @staticmethod
    def _extension_attributes(
        connection: sqlite3.Connection, table: str, identifier: str
    ) -> dict[str, object]:
        available = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,)
            )
        }
        if table not in available:
            return {}
        row = connection.execute(
            f'SELECT * FROM "{table}" WHERE ocel_id = ?', (identifier,)
        ).fetchone()
        if row is None:
            return {}
        return {
            key: row[key]
            for key in row.keys()
            if key != "ocel_id"
        }