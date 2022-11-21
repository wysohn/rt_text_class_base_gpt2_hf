from ..misc.config import *

ID_COL_KEY = 'idField'
TEXT_COL_KEY = 'documentField'
Y_COL_KEY = 'targetField'


class DataSchema:
    def __init__(self, schema: dict):
        self._schema = schema

    def col_id_key(self) -> str:
        return self._schema['inputDatasets'][SCHEMA_TYPE][ID_COL_KEY]

    def col_text_key(self) -> str:
        return self._schema['inputDatasets'][SCHEMA_TYPE][TEXT_COL_KEY]

    def col_label_key(self) -> str:
        return self._schema['inputDatasets'][SCHEMA_TYPE][Y_COL_KEY]
