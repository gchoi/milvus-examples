import os
from collections import defaultdict
from typing import List, Literal

from dotenv import load_dotenv
from tqdm import tqdm
import milvus
import numpy as np
from scipy.sparse import csr_matrix
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

from milvus.utils import get_configurations
from milvus.model import Model
from milvus.conf import Logger
from milvus.client.utils import (
    create_collection,
    insert,
    search
)

from dataset import nano_dataset


# -- logger settings
logger = Logger(env="dev")

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def insert_data(
    uri: str,
    batch_size: int,
    collection_name: str,
    model: milvus.model.model.Model,
    text_list: List[str]
) -> None:
    for row_id in tqdm(range(0, len(text_list), batch_size), desc="Inserting"):
        print("\n")
        logger.info(f"Inserting batch {row_id} to {row_id + batch_size} for collection '{collection_name}'.")
        batch_texts = text_list[row_id : row_id + batch_size]

        batch_embeddings = []
        for i, batch_text in enumerate(tqdm(batch_texts, desc="Creating embeddings", position=0)):
            batch_embeddings.append(model.get_text_embedding(text=batch_text))

        batch_ids = [row_id + j for j in range(len(batch_texts))]
        batch_data = [
            {
                "id": id_,
                "text": text,
                "vector": vector,
            }
            for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
        ]
        insert(uri=uri, collection_name=collection_name, data=batch_data)
    return


def rerank_relations(
    query: str,
    model: Literal[ChatOpenAI,],
    relation_candidate_texts: list[str],
    relation_candidate_ids: list[str],
    query_prompt_one_shot_input,
    query_prompt_one_shot_output,
    query_prompt_template
) -> List[int]:
    relation_des_str = "\n".join(
        map(
            lambda item: f"[{item[0]}] {item[1]}",
            zip(relation_candidate_ids, relation_candidate_texts),
        )
    ).strip()
    rerank_prompts = ChatPromptTemplate.from_messages(
        [
            HumanMessage(query_prompt_one_shot_input),
            AIMessage(query_prompt_one_shot_output),
            HumanMessagePromptTemplate.from_template(query_prompt_template),
        ]
    )
    rerank_chain = (
        rerank_prompts
        | model.bind(response_format={"type": "json_object"})
        | JsonOutputParser()
    )
    rerank_res = rerank_chain.invoke(
        {
            "question": query,
            "relation_des_str": relation_des_str
        }
    )
    rerank_relation_ids = []
    rerank_relation_lines = rerank_res["useful_relationships"]
    id_2_lines = {}
    for line in rerank_relation_lines:
        id_ = int(line[line.find("[") + 1 : line.find("]")])
        id_2_lines[id_] = line.strip()
        rerank_relation_ids.append(id_)
    return rerank_relation_ids


def main():
    ########################################################################
    # Configurations
    ########################################################################

    # -- Get configurations
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_graph_rag.yaml")
    configs = get_configurations(config_yaml_path=config_path)

    # -- Milvus configurations
    uri = f"{configs.get('milvus').get('host')}:{configs.get('milvus').get('port')}"
    if not uri.startswith("http://"):
        uri = f"http://{uri}"

    # -- Model configurations
    model = Model(
        platform=configs.get("model").get("platform"),
        embedding_model=configs.get("model").get("embedding_model"),
        chat_model=configs.get("model").get("chat_model"),
    )
    chat_model, _ = model.get_models(temperature=0)


    ########################################################################
    # Offline Data Pre-processing
    ########################################################################

    entityid_2_relationids = defaultdict(list)
    relationid_2_passageids = defaultdict(list)

    entities = []
    relations = []
    passages = []
    for passage_id, dataset_info in enumerate(nano_dataset):
        passage, triplets = dataset_info["passage"], dataset_info["triplets"]
        passages.append(passage)
        for triplet in triplets:
            if triplet[0] not in entities:
                entities.append(triplet[0])
            if triplet[2] not in entities:
                entities.append(triplet[2])
            relation = " ".join(triplet)
            if relation not in relations:
                relations.append(relation)
                entityid_2_relationids[entities.index(triplet[0])].append(len(relations) - 1)
                entityid_2_relationids[entities.index(triplet[2])].append(len(relations) - 1)
            relationid_2_passageids[relations.index(relation)].append(passage_id)


    ########################################################################
    # Data Insertion
    ########################################################################

    embedding_dim = len(model.get_text_embedding(text="foo"))

    # -- Create collections
    batch_size = configs.get("model").get("batch_size")

    # Entity collection
    create_collection(
        uri=uri,
        collection_name=configs.get("milvus").get("entity_collection_name"),
        embedding_dim=embedding_dim,
        consistency_level="Bounded",
        overwrite=True,
        collection_type="semantic_search",
        dense_search_metric_type=configs.get("milvus").get("search").get("metric_type"),
    )
    insert_data(
        uri=uri,
        batch_size=batch_size,
        collection_name=configs.get("milvus").get("entity_collection_name"),
        model=model,
        text_list=entities
    )

    # Relation collection
    create_collection(
        uri=uri,
        collection_name=configs.get("milvus").get("relation_collection_name"),
        embedding_dim=embedding_dim,
        consistency_level="Bounded",
        overwrite=True,
        collection_type="semantic_search",
        dense_search_metric_type=configs.get("milvus").get("search").get("metric_type"),
    )
    insert_data(
        uri=uri,
        batch_size=batch_size,
        collection_name=configs.get("milvus").get("relation_collection_name"),
        model=model,
        text_list=relations
    )

    # Passage collection
    create_collection(
        uri=uri,
        collection_name=configs.get("milvus").get("passage_collection_name"),
        embedding_dim=embedding_dim,
        consistency_level="Bounded",
        overwrite=True,
        collection_type="semantic_search",
        dense_search_metric_type=configs.get("milvus").get("search").get("metric_type"),
    )
    insert_data(
        uri=uri,
        batch_size=batch_size,
        collection_name=configs.get("milvus").get("passage_collection_name"),
        model=model,
        text_list=passages
    )


    ########################################################################
    # Similarity Retrieval
    ########################################################################

    query = "What contribution did the son of Euler's teacher make?"
    query_ner_list = ["Euler"]
    top_k = configs.get("milvus").get("search").get("limit")

    query_ner_embeddings = [model.get_text_embedding(text=query_ner) for query_ner in query_ner_list]

    entity_search_res = search(
        uri=uri,
        collection_name=configs.get("milvus").get("entity_collection_name"),
        queries=[query],
        query_embeddings=query_ner_embeddings,
        search_type="semantic_search",
        limit=top_k,
        dense_search_metric_type=configs.get("milvus").get("search").get("metric_type"),
        output_fields=["id"]
    )

    query_embedding = model.get_text_embedding(text=query)

    relation_search_res = search(
        uri=uri,
        collection_name=configs.get("milvus").get("relation_collection_name"),
        queries=[query],
        query_embeddings=[query_embedding],
        search_type="semantic_search",
        limit=top_k,
        dense_search_metric_type=configs.get("milvus").get("search").get("metric_type"),
        output_fields=["id"]
    )[0]


    ########################################################################
    # Expand Subgraph
    ########################################################################

    # Construct the adjacency matrix of entities and relations where the value of the adjacency matrix is 1
    # if an entity is related to a relation, otherwise 0.
    entity_relation_adj = np.zeros((len(entities), len(relations)))
    for entity_id, entity in enumerate(entities):
        entity_relation_adj[entity_id, entityid_2_relationids[entity_id]] = 1

    # Convert the adjacency matrix to a sparse matrix for efficient computation.
    entity_relation_adj = csr_matrix(entity_relation_adj)

    # Use the entity-relation adjacency matrix to construct 1 degree entity-entity and relation-relation
    # adjacency matrices.
    entity_adj_1_degree = entity_relation_adj @ entity_relation_adj.T
    relation_adj_1_degree = entity_relation_adj.T @ entity_relation_adj

    # Specify the target degree of the subgraph to be expanded.
    # 1 or 2 is enough for most cases.
    target_degree = 1

    # Compute the target degree adjacency matrices using matrix multiplication.
    entity_adj_target_degree = entity_adj_1_degree
    for _ in range(target_degree - 1):
        entity_adj_target_degree = entity_adj_target_degree * entity_adj_1_degree
    relation_adj_target_degree = relation_adj_1_degree
    for _ in range(target_degree - 1):
        relation_adj_target_degree = relation_adj_target_degree * relation_adj_1_degree

    entity_relation_adj_target_degree = entity_adj_target_degree @ entity_relation_adj

    expanded_relations_from_relation = set()
    expanded_relations_from_entity = set()
    # You can set the similarity threshold here to guarantee the quality of the retrieved ones.
    # entity_sim_filter_thresh = ...
    # relation_sim_filter_thresh = ...

    filtered_hit_relation_ids = [
        relation_res["entity"]["id"]
        for relation_res in relation_search_res
        # if relation_res['distance'] > relation_sim_filter_thresh
    ]
    for hit_relation_id in filtered_hit_relation_ids:
        expanded_relations_from_relation.update(relation_adj_target_degree[hit_relation_id].nonzero()[1].tolist())

    filtered_hit_entity_ids = [
        one_entity_res["entity"]["id"]
        for one_entity_search_res in entity_search_res
        for one_entity_res in one_entity_search_res
        # if one_entity_res['distance'] > entity_sim_filter_thresh
    ]

    for filtered_hit_entity_id in filtered_hit_entity_ids:
        expanded_relations_from_entity.update(
            entity_relation_adj_target_degree[filtered_hit_entity_id].nonzero()[1].tolist()
        )

    # Merge the expanded relations from the relation and entity retrieval ways.
    relation_candidate_ids = list(expanded_relations_from_relation | expanded_relations_from_entity)
    relation_candidate_texts = [relations[relation_id] for relation_id in relation_candidate_ids]


    ########################################################################
    # LLM Reranking
    ########################################################################

    query_prompt_one_shot_input = """
    I will provide you with a list of relationship descriptions. Your task is to select 3 relationships that may be 
    useful to answer the given question. Please return a JSON object containing your thought process and a list of 
    the selected relationships in order of their relevance.

    Question:
    When was the mother of the leader of the Third Crusade born?

    Relationship descriptions:
    [1] Eleanor was born in 1122.
    [2] Eleanor married King Louis VII of France.
    [3] Eleanor was the Duchess of Aquitaine.
    [4] Eleanor participated in the Second Crusade.
    [5] Eleanor had eight children.
    [6] Eleanor was married to Henry II of England.
    [7] Eleanor was the mother of Richard the Lionheart.
    [8] Richard the Lionheart was the King of England.
    [9] Henry II was the father of Richard the Lionheart.
    [10] Henry II was the King of England.
    [11] Richard the Lionheart led the Third Crusade.

    """
    query_prompt_one_shot_output = """
    {"thought_process": "To answer the question about the birth of the mother of the leader of the Third Crusade, 
    I first need to identify who led the Third Crusade and then determine who his mother was. After identifying 
    his mother, I can look for the relationship that mentions her birth.", "useful_relationships": ["[11] Richard 
    the Lionheart led the Third Crusade", "[7] Eleanor was the mother of Richard the Lionheart", "[1] Eleanor was born 
    in 1122"]}
    """

    query_prompt_template = """
    Question: {question}

    Relationship descriptions: {relation_des_str}
    """

    rerank_relation_ids = rerank_relations(
        query=query,
        model=chat_model,
        relation_candidate_texts=relation_candidate_texts,
        relation_candidate_ids=relation_candidate_ids,
        query_prompt_one_shot_input=query_prompt_one_shot_input,
        query_prompt_one_shot_output=query_prompt_one_shot_output,
        query_prompt_template=query_prompt_template
    )


    ########################################################################
    # Get Final Results
    ########################################################################

    final_top_k = 2

    final_passages = []
    final_passage_ids = []
    for relation_id in rerank_relation_ids:
        for passage_id in relationid_2_passageids[relation_id]:
            if passage_id not in final_passage_ids:
                final_passage_ids.append(passage_id)
                final_passages.append(passages[passage_id])
    passages_from_graph_rag = final_passages[:final_top_k]

    naive_passage_res = search(
        uri=uri,
        collection_name=configs.get("milvus").get("passage_collection_name"),
        queries=[query_embedding],
        query_embeddings=query_ner_embeddings,
        search_type="semantic_search",
        limit=final_top_k,
        dense_search_metric_type=configs.get("milvus").get("search").get("metric_type"),
        output_fields=["text"],
    )[0]
    passages_from_naive_rag = [res["entity"]["text"] for res in naive_passage_res]

    print(
        f"Passages retrieved from naive RAG: \n{passages_from_naive_rag}\n\n"
        f"Passages retrieved from Graph RAG: \n{passages_from_graph_rag}\n\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """
                Use the following pieces of retrieved context to answer the question. If there is not enough 
                information in the retrieved context to answer the question, just say that you don't know.
                
                Question: {question}
                Context: {context}
                Answer:
                """,
            )
        ]
    )

    rag_chain = prompt | chat_model | StrOutputParser()

    answer_from_naive_rag = rag_chain.invoke(
        {
            "question": query,
            "context": "\n".join(passages_from_naive_rag)
        }
    )
    answer_from_graph_rag = rag_chain.invoke(
        {
            "question": query,
            "context": "\n".join(passages_from_graph_rag)
        }
    )

    print(f"Question: {query}")
    print(f"Answer from naive RAG: {answer_from_naive_rag}")
    print(f"Answer from Graph RAG: {answer_from_graph_rag}")
    return


if __name__ == "__main__":
    main()
