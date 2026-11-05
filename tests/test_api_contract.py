import main
from fastapi.testclient import TestClient
from pydantic import ValidationError

VALID_PAYLOAD = {
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "A",
    "loan_amnt": 10000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 10000 / 60000,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5,
}


def test_feature_columns_are_defined():
    assert main.FEATURE_COLUMNS[0] == "person_age"
    assert len(main.FEATURE_COLUMNS) == 11


def test_loan_application_rejects_unknown_category():
    payload = {**VALID_PAYLOAD, "person_home_ownership": "CASTLE"}
    try:
        main.LoanApplication(**payload)
    except ValidationError:
        return
    raise AssertionError("Invalid home ownership category should be rejected")


def test_loan_application_accepts_valid_payload():
    result = main.LoanApplication(**VALID_PAYLOAD)
    assert result.loan_grade == "A"


def test_predict_returns_valid_probability():
    with TestClient(main.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        probability = response.json()["default_probability"]
        assert 0.0 <= probability <= 1.0