def test_tiny(init_database, client):
    """This test case mirrors the tiny_na test case in persephone,
    but done just through API calls, it uses real Yongning Na audio
    and transcriptions."""
    import json
    import os
    def upload_audio(path: str, filename: str) -> int:
        """Helper to upload a file, where `path` is the directory
        returns ID of uploaded file
        """
        with open(os.path.join(path, filename), "rb") as audio_file:
            data = {'audioFile': (audio_file, filename)}
            response = client.post(
                ('/v0.1/audio'),
                data=data,
                content_type='multipart/form-data'
            )
        assert response.status_code == 201
        response_data = json.loads(response.data.decode('utf8'))
        assert response_data['id']
        return response_data['id']

    def upload_transcription(path: str, filename: str) -> int:
        """Helper to upload a transcription file, where `path` is the directory
        returns ID of uploaded file
        """
        with open(os.path.join(path, filename), "rb") as transcription_file:
            data = {'transcriptionFile': (transcription_file, filename)}
            response = client.post(
                ('/v0.1/transcription/fromFile'),
                data=data,
                content_type='multipart/form-data'
            )
        assert response.status_code == 201
        response_data = json.loads(response.data.decode('utf8'))
        assert response_data['id']
        return response_data['id']

    def create_utterance(audio_id: int, transcription_id: int) -> int:
        """Helper to create an utterance, returns id of utterance created"""
        data = {
            "audioId": audio_id,
            "transcriptionId": transcription_id
        }

        response = client.post(
            '/v0.1/utterance',
            data=json.dumps(data),
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == 201, response
        response_data = json.loads(response.data.decode('utf8'))
        assert response_data['id']
        return response_data['id']

    data_path = "tests/na_tiny_example_files"

    train_prefixes = [
        "crdo-NRU_NUMPLUSCL_L2_TIMES_1TO100_F4_24SEPT2011_AUDIOPLUSEGG.53",
        "crdo-NRU_NUMPLUSCL_L2_TIMES_1TO100_F4_24SEPT2011_AUDIOPLUSEGG.54"
    ]
    validation_prefixes = [
        "crdo-NRU_NUMPLUSCL_L2_TIMES_1TO100_F4_24SEPT2011_AUDIOPLUSEGG.55"
    ]
    test_prefixes = [
        "crdo-NRU_NUMPLUSCL_L2_TIMES_1TO100_F4_24SEPT2011_AUDIOPLUSEGG.56"
    ]

    train_audio_ids = [
        upload_audio(data_path, train_prefixes[0]+".wav"),
        upload_audio(data_path, train_prefixes[1]+".wav")
    ]
    validation_audio_ids = [
        upload_audio(data_path, validation_prefixes[0]+".wav")
    ]
    test_audio_ids = [
        upload_audio(data_path, test_prefixes[0]+".wav")
    ]

    train_transcriptions_ids = [
        upload_transcription(data_path, train_prefixes[0]+".phonemes"),
        upload_transcription(data_path, train_prefixes[1]+".phonemes")
    ]
    validation_transcriptions_ids = [
        upload_transcription(data_path, validation_prefixes[0]+".phonemes")
    ]
    test_transcriptions_ids = [
        upload_transcription(data_path, test_prefixes[0]+".phonemes")
    ]

    training_utterances = [
        create_utterance(train_audio_ids[0], train_transcriptions_ids[0]),
        create_utterance(train_audio_ids[1], train_transcriptions_ids[1]),
    ]

    validation_utterances = [
        create_utterance(validation_audio_ids[0], validation_transcriptions_ids[0])
    ]

    test_utterances = [
        create_utterance(test_audio_ids[0], test_transcriptions_ids[0])
    ]

    data = {
        "name": "Test Na tiny corpus",
        "labelType": "phonemes",
        "featureType": "fbank",
        "preprocessed": "false",
        "testing": test_utterances,
        "training": training_utterances,
        "validation": validation_utterances
    }

    response = client.post(
        '/v0.1/corpus',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 201

    corpus_response_data = json.loads(response.data.decode('utf8'))
    corpus_id = corpus_response_data['id']

    model_data = {
        "name": "Test Na tiny model",
        "beamWidth": 1,
        "corpusID": corpus_id,
        "decodingMergeRepeated": True,
        "earlyStoppingSteps": 1,
        "numberLayers": 2,
        "hiddenSize": 250,
        "maximumEpochs": 5,
        "minimumEpochs": 2,
    }

    response = client.post(
        '/v0.1/model',
        data=json.dumps(model_data),
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 201
    model_response_data = json.loads(response.data.decode('utf8'))
    model_id = model_response_data['id']

    response = client.post(
        '/v0.1/model/train/{}'.format(model_id),
        data=json.dumps(model_data),
        headers={'Content-Type': 'application/json'}
    )

    assert response.status_code >= 200 and response.status_code < 300

    transcribe_test_audio_prefix = "crdo-NRU_NUMPLUSCL_L2_TIMES_1TO100_F4_24SEPT2011_AUDIOPLUSEGG.58"
    audio_to_transcribe_id = upload_audio(data_path, transcribe_test_audio_prefix+".wav")

    response = client.post(
        '/v0.1/model/transcribe/{}/{}'.format(model_id, audio_to_transcribe_id),
        headers={'Content-Type': 'application/json'}
    )

    assert response.status_code == 201
