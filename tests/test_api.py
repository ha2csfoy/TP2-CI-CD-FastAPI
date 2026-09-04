from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_prediction_correcte():
    """Valide une prédiction correcte avec [1.0, 2.0, 3.0]."""
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [2.0, 4.0, 6.0]
    }


def test_prediction_incorrecte():
    """
    Vérifie que la réponse obtenue ne correspond pas
    à un résultat volontairement faux.
    """
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )

    resultat_volontairement_faux = {
        "predictions": [3.0, 6.0, 9.0]
    }

    assert response.status_code == 200
    assert response.json() != resultat_volontairement_faux


def test_json_incorrect():
    """Vérifie le rejet d'un JSON dans lequel le champ features manque."""
    response = client.post(
        "/predict",
        json={
            "feature1": 3.5,
            "feature2": 1.2,
            "feature3": 4.9,
        },
    )

    assert response.status_code == 422
    assert "detail" in response.json()
