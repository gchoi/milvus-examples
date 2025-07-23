import re
import textwrap
from typing import Any, Dict, List

from langchain_core.documents import Document


def _parse_column_descriptions(column_description: str
                               ) -> Dict[str, Dict[str, Any]]:
    """Parses the column_description string into a dictionary."""

    columns = {}
    # "Column "으로 시작하는 블록 단위 분리하여 columns 딕셔너리 생성
    column_blocks = re.split(r"\n(?=Column )", column_description.strip())
    for block in column_blocks:
        # col_name 저장
        lines = block.strip().split("\n")
        if not lines:
            continue
        match = re.match(r"Column (\w+)", lines[0])
        if not match:
            continue
        col_name = match.group(1)

        # col_info 저장
        col_info = {}
        for line in lines[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                value = value.strip()
                if key.strip() == "Examples":  # "Examples"는 리스트로 되어있어서 따로 처리
                    try:
                        value = eval(value)
                    except Exception:
                        pass
                col_info[key.strip()] = value

        columns[col_name] = col_info

    return columns


def td2docs_table_comment(
    td: Dict[str, str],
) -> List[Document]:
    docs = []
    for table_name, info in td.items():
        table_comment = info["table_comment"]
        page_content = f"""
        ### Table Name
        {table_name}

        #### Table Comment
        {table_comment}
        """
        page_content = textwrap.dedent(page_content)
        docs.append(
            Document(page_content=page_content,
                     metadata={"table_name": table_name})
        )
    return docs


def td2docs_column_level(td: Dict[str, Dict[str, str]]) -> List[Document]:
    """Convert table description to documents at the column level."""
    docs = []
    for table_name, info in td.items():
        column_description_text = info.get("column_description", "")
        column_info_dict = _parse_column_descriptions(column_description_text)

        for column_name, desc_dict in column_info_dict.items():
            desc_lines = [f"- {k}: {v}" for k, v in desc_dict.items()]
            desc_text = "\n".join(desc_lines)

            page_content = f"""
            ### Table Name
            {table_name}

            ### Column Name
            {column_name}

            #### Column Description
            {desc_text}
            """
            docs.append(
                Document(
                    page_content=textwrap.dedent(page_content).strip(),
                    metadata={"table_name": table_name,
                              "column_name": column_name},
                )
            )
    return docs


if __name__ == "__main__":
    from .preprocess import load_td, preprocess_td

    json_path = (
        json_path
    ) = "experiments/data/_generated_schemas_shilla_t2s_devloper_gemma3_12b_20250604.json"
    data = load_td(json_path)
    td = preprocess_td(data)
    docs = td2docs_column_level(td)
    print(docs)
