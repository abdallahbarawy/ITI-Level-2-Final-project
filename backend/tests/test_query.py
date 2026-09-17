import pytest
from fastapi.testclient import TestClient

from app import main
from app.core.config import Settings


class StubGeneration:
    def __init__(self, settings, config):
        self.model = config["ollama_model"]

    def check_connection(self):
        pass

    def answer(self, question, chunks):
        assert chunks
        assert chunks[0]["source"] == "offside_trap.txt"
        assert "offside" in chunks[0]["text"].lower()
        return "Defenders step forward together; a failed trap leaves space behind. [Source: offside_trap.txt]"

    def close(self):
        pass


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(main, "GenerationService", StubGeneration)
        app = main.create_app(Settings(_env_file=None))
        with TestClient(app) as test_client:
            yield test_client


def test_query_happy_path(client):
    response = client.post(
        "/query", json={"question": "What is the offside trap and what is its biggest risk?"}
    )
    assert response.status_code == 200
    result = response.json()
    assert "space behind" in result["answer"]
    assert "offside_trap.txt" in result["sources"]
    assert len(result["sources"]) == len(set(result["sources"]))


def test_query_invalid_input(client):
    for payload in ({}, {"question": ""}, {"question": "   "}, {"question": 123}):
        response = client.post("/query", json=payload)
        assert response.status_code == 422
