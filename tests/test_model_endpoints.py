"""Tests for model related endpoints"""

def test_invalid_min_max_steps(init_database, client, create_corpus):
    corpus_id = create_corpus()
    data = {
        "corpusID": corpus_id,
        "maximumEpochs": 10,
        "minimumEpochs": 100,
        "name": "Bad mix max model"
    }

    import json
    response = client.post(
        '/v0.1/model',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )

    assert response.status_code == 400

def test_invalid_epoch_steps(init_database, client, create_corpus):
    import json
    corpus_id = create_corpus()
    data = {
        "corpusID": corpus_id,
        "maximumEpochs": 5,
        "minimumEpochs": -1,
        "name": "Bad minimum epoch"
    }

    response = client.post(
        '/v0.1/model',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )

    assert response.status_code == 400

    data = {
        "corpusID": corpus_id,
        "maximumEpochs": -10,
        "minimumEpochs": 5,
        "name": "Bad maximum epoch"
    }

    response = client.post(
        '/v0.1/model',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 400


def test_invalid_early_stopping(init_database, client, create_corpus):
    corpus_id = create_corpus()
    import json
    data = {
        "corpusID": corpus_id,
        "earlyStoppingSteps": -1,
        "name": "Bad maximum epoch"
    }

    response = client.post(
        '/v0.1/model',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 400

def test_invalid_LER(init_database, client, create_corpus):
    """Tests that bogus values for maximum error rates are rejected"""
    corpus_id = create_corpus()
    import json

    data = {
        "corpusID": corpus_id,
        "maximumTrainingLER": -1,
        "name": "Bad maximum epoch"
    }

    response = client.post(
        '/v0.1/model',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 400

    data = {
        "corpusID": corpus_id,
        "maximumValidationLER": -1,
        "name": "Bad maximum epoch"
    }
    response = client.post(
        '/v0.1/model',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 400

def test_create_model(init_database, client, create_corpus):
    """Test that we can create a model from the API"""
    import json
    corpus_id = create_corpus()

    model_data = {
        "name": "Test model",
        "beamWidth": 1,
        "corpusID": corpus_id,
        "decodingMergeRepeated": True,
        "earlyStoppingSteps": 1,
        "numberLayers": 2,
        "hiddenSize": 2,
        "maximumEpochs": 2,
        "minimumEpochs": 1,
    }

    response = client.post(
        '/v0.1/model',
        data=json.dumps(model_data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 201

def test_create_model_no_corpus(init_database, client):
    """Test that we creation of a model from the API with an invalid corpus produces a 400 error"""
    import json
    corpus_id = 9999999 # shouldn't exist

    model_data = {
        "name": "Test model",
        "beamWidth": 1,
        "corpusID": corpus_id,
        "decodingMergeRepeated": True,
        "earlyStoppingSteps": 1,
        "numberLayers": 2,
        "hiddenSize": 2,
        "maximumEpochs": 2,
        "minimumEpochs": 1,
    }

    response = client.post(
        '/v0.1/model',
        data=json.dumps(model_data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 400

def test_get_model(init_database, client, create_corpus):
    """Test that we can create a model from the API and then retrieve it"""
    import json
    corpus_id = create_corpus()

    model_data = {
        "name": "Test model",
        "beamWidth": 1,
        "corpusID": corpus_id,
        "decodingMergeRepeated": True,
        "earlyStoppingSteps": 1,
        "numberLayers": 2,
        "hiddenSize": 2,
        "maximumEpochs": 2,
        "minimumEpochs": 1,
        "maximumTrainingLER": 0.4,
        "maximumValidationLER": 0.8,
    }

    response = client.post(
        '/v0.1/model',
        data=json.dumps(model_data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 201
    model_response_data = json.loads(response.data.decode('utf8'))
    model_id = model_response_data['id']

    assert 'maximumTrainingLER' in model_response_data, model_response_data
    assert model_response_data['maximumTrainingLER'] == 0.4
    assert model_response_data['maximumValidationLER'] == 0.8

    response = client.get(
        '/v0.1/model/{}'.format(model_id),
    )
    assert response.status_code == 200
    model_get_data = json.loads(response.data.decode('utf8'))

    assert model_get_data["minimumEpochs"] == 1
    assert model_get_data["maximumEpochs"] == 2
    assert model_get_data["maximumTrainingLER"] == 0.4
    assert model_get_data["maximumValidationLER"] == 0.8