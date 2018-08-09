def test_invalid_min_max_steps(client):
    data = {
        "corpusID": 1,
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

def test_invalid_epoch_steps(client):
    import json
    data = {
        "corpusID": 1,
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
        "corpusID": 1,
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


def test_invalid_early_stopping(client):
    import json
    data = {
        "corpusID": 1,
        "earlyStoppingSteps": -1,
        "name": "Bad maximum epoch"
    }

    response = client.post(
        '/v0.1/model',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 400


def create_model():
    """Test that we can create a model from the API"""
     # Create mock audio uploads
    response = upload_audio(create_sine(note="A"), filename="a.wav")
    assert response.status_code == 201
    wav_response_data = json.loads(response.data.decode('utf8'))
    wav_id_a = wav_response_data['id']

    response = upload_audio(create_sine(note="B"), filename="b.wav")
    assert response.status_code == 201
    wav_response_data = json.loads(response.data.decode('utf8'))
    wav_id_b = wav_response_data['id']

    response = upload_audio(create_sine(note="C"), filename="c.wav")
    assert response.status_code == 201
    wav_response_data = json.loads(response.data.decode('utf8'))
    wav_id_c = wav_response_data['id']

    # Create mock transcription uploads
    response = upload_transcription("A", filename="a.phonemes")
    assert response.status_code == 201
    transcription_response_data = json.loads(response.data.decode('utf8'))
    transcription_id_a = transcription_response_data['id']

    response = upload_transcription("B", filename="b.phonemes")
    assert response.status_code == 201
    transcription_response_data = json.loads(response.data.decode('utf8'))
    transcription_id_b = transcription_response_data['id']

    response = upload_transcription("C", filename="c.phonemes")
    assert response.status_code == 201
    transcription_response_data = json.loads(response.data.decode('utf8'))
    transcription_id_c = transcription_response_data['id']

    data = {
        "audioId": wav_id_a,
        "transcriptionId": transcription_id_a
    }

    response = client.post(
        '/v0.1/utterance',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 201
    utterance_response_data = json.loads(response.data.decode('utf8'))
    utterance_id_a = utterance_response_data['id']

    data = {
        "audioId": wav_id_b,
        "transcriptionId": transcription_id_b
    }

    response = client.post(
        '/v0.1/utterance',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 201
    utterance_response_data = json.loads(response.data.decode('utf8'))
    utterance_id_b = utterance_response_data['id']

    data = {
        "audioId": wav_id_c,
        "transcriptionId": transcription_id_c
    }

    response = client.post(
        '/v0.1/utterance',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 201
    utterance_response_data = json.loads(response.data.decode('utf8'))
    utterance_id_c = utterance_response_data['id']

    corpus_data = {
        "name": "Test Corpus",
        "label_type": "phonemes",
        "feature_type": "fbank",
        "preprocessed": "false",
        "testing": [
            utterance_id_a
        ],
        "training": [
            utterance_id_b
        ],
        "validation": [
            utterance_id_c
        ]
    }

    response = client.post(
        '/v0.1/corpus',
        data=json.dumps(corpus_data),
        headers={'Content-Type': 'application/json'}
    )

    assert response.status_code == 201

    corpus_response_data = json.loads(response.data.decode('utf8'))
    corpus_id = corpus_response_data['id']

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

