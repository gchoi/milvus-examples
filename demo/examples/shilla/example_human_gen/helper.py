from typing import Dict
import logging
import os
import pandas as pd
import json
from pathlib import Path, PosixPath


def find_project_root(marker: str=".git")-> PosixPath:
    """Find project root path
    marker 가 존재하는 path를 반환

    Parameters
    ----------
        marker: str = ".git"

    Returns
    -------
        Root path: PosixPath
    """
    current = Path(__file__).resolve().parent
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find {marker} in any parent directory.")


def load_tables_description(db_directory_path: str, use_value_description: bool) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Loads table descriptions from CSV files in the database directory.

    Args:
        db_directory_path (str): The path to the database directory.
        use_value_description (bool): Whether to include value descriptions.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary containing table descriptions.
    """
    encoding_types = ['utf-8-sig', 'cp1252']
    description_path = Path(db_directory_path) / "database_description"
    
    if not description_path.exists():
        logging.warning(f"Description path does not exist: {description_path}")
        return {}
    
    table_description = {}
    for csv_file in description_path.glob("*.csv"):
        table_name = csv_file.stem.lower().strip()
        table_description[table_name] = {}
        could_read = False
        for encoding_type in encoding_types:
            try:
                table_description_df = pd.read_csv(csv_file, index_col=False, encoding=encoding_type)
                for _, row in table_description_df.iterrows():
                    column_name = row['original_column_name']
                    expanded_column_name = row.get('column_name', '').strip() if pd.notna(row.get('column_name', '')) else ""
                    column_description = row.get('column_description', '').replace('\n', ' ').replace("commonsense evidence:", "").strip() if pd.notna(row.get('column_description', '')) else ""
                    data_format = row.get('data_format', '').strip() if pd.notna(row.get('data_format', '')) else ""
                    value_description = ""
                    if use_value_description and pd.notna(row.get('value_description', '')):
                        value_description = row['value_description'].replace('\n', ' ').replace("commonsense evidence:", "").strip()
                        if value_description.lower().startswith("not useful"):
                            value_description = value_description[10:].strip()
                    
                    table_description[table_name][column_name.lower().strip()] = {
                        "original_column_name": column_name,
                        "column_name": expanded_column_name,
                        "column_description": column_description,
                        "data_format": data_format,
                        "value_description": value_description
                    }
                logging.info(f"Loaded descriptions from {csv_file} with encoding {encoding_type}")
                could_read = True
                break
            except Exception:
                continue
    if not could_read:
        logging.warning(f"Could not read descriptions from {csv_file}")
    return table_description
