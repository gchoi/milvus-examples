from milvus.preprocess.preprocess import load_td



def test_load_td():
    path = "tests/data/data.json"
    data = load_td(path=path)
    assert isinstance(data, dict)


if __name__=="__main__":
    test_load_td()