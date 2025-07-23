from milvus.preprocess.preprocess import preprocess_td

data = {
    "db_info": "It's a test db.",
    "tables": [
        {
            "table_name": "table1",
            "table_comment": "table1 is very good.",
            "fields": [
                {
                    "field_name": "critn_yyyymm",
                    "field_type": "VARCHAR(6)",
                    "field_comment": "기준년월",
                    "field_primary_key": True,
                    "field_default": None,
                    "field_unique": False,
                    "field_nullable": False,
                    "field_autoincrement": False,
                    "field_examples": [
                        "202505",
                        "202503",
                        "202504"
                    ],
                    "total_count": 14377,
                    "distinct_count": 3,
                    "max_value": None,
                    "min_value": None,
                    "avg_value": None,
                    "char_len_max": 6,
                    "char_len_min": None,
                    "category": "Enum",
                    "dim_or_meas": "Dimension",
                    "date_min_gran": "",
                    "full_en_col_name": "reference_date",
                    "ko_col_name": "기준일"
                }
            ]
        },
        {
            "table_name": "table2",
            "table_comment": "table2 is very bad.",
            "fields": [
                {
                    "field_name": "critn_yyyymm",
                    "field_type": "VARCHAR(6)",
                    "field_comment": "기준년월",
                    "field_primary_key": True,
                    "field_default": None,
                    "field_unique": False,
                    "field_nullable": False,
                    "field_autoincrement": False,
                    "field_examples": [
                        "202505",
                        "202503",
                        "202504"
                    ],
                    "total_count": 14377,
                    "distinct_count": 3,
                    "max_value": None,
                    "min_value": None,
                    "avg_value": None,
                    "char_len_max": 6,
                    "char_len_min": None,
                    "category": "Enum",
                    "dim_or_meas": "Dimension",
                    "date_min_gran": "",
                    "full_en_col_name": "reference_date",
                    "ko_col_name": "기준일"
                },
                {
                    "field_name": "line_cd",
                    "field_type": "VARCHAR",
                    "field_comment": "공정코드",
                    "field_primary_key": True,
                    "field_default": None,
                    "field_unique": False,
                    "field_nullable": False,
                    "field_autoincrement": False,
                    "field_examples": [
                        "10",
                        "11",
                        "12",
                        "13",
                        "14"
                    ],
                    "total_count": 14377,
                    "distinct_count": 12,
                    "max_value": None,
                    "min_value": None,
                    "avg_value": None,
                    "char_len_max": 3,
                    "char_len_min": None,
                    "category": "Enum",
                    "dim_or_meas": "Dimension",
                    "date_min_gran": "",
                    "full_en_col_name": "process_code",
                    "ko_col_name": "공정ID"
                }
            ]
        }
    ]
}


def test_preprocess_td():
    td = preprocess_td(data=data)
    
    assert len(td) == 2  # table 갯수
    assert td["table1"]["table_comment"] == "table1 is very good."
    assert isinstance(td["table2"]["column_description"], str)