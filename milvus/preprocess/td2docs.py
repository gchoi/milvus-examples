from typing import List, Dict
import textwrap

from langchain_core.documents import Document


def td2docs_v1(
    td: Dict[str, str],
)-> List[Document]:
    """Convert Table Description to Documents

    v1: table comment와 column descriptions을 함께 page_content에 넣는 방법.

    Parameters
    ----------
        td: Dict[str, str]
            The preprocessed table description.
    
    Returns
    -------
        List[Document]: Documents of langchian format.
    """
    docs = []
    for table_name, info in td.items():
        table_comment = info["table_comment"]
        column_descriptions = info["column_description"]
        page_content = f"""
        ### Table Name
        {table_name}

        #### Table Comment
        {table_comment}

        ### Column Descriptions
        {column_descriptions}\n\n
        """
        page_content = textwrap.dedent(page_content)
        docs.append(Document(page_content=page_content, metadata={"table_name": table_name}))
    return docs


if __name__=="__main__":
    from .preprocess import load_td, preprocess_td

    json_path = "shilla/experiments/data/_generated_schemas_shilla_t2s_devloper_gemma3_12b_20250528_shilla_t2s_devloper_gemma3_12b_20250528.json"
    data = load_td(json_path)
    td = preprocess_td(data)
    docs = td2docs_v1(td)
    print(docs)