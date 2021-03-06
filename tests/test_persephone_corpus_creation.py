from unittest.mock import MagicMock

def test_create_corpus_file_structure(tmpdir):
    """Test filesystem tasks related to corpus creation"""
    from persephone_api.api_endpoints.corpus import create_corpus_file_structure
    from pathlib import Path
    corpus_base_dir = Path(str(tmpdir.mkdir("corpus_base")))
    audio_uploads_dir = Path(str(tmpdir.mkdir("audio_uploads")))
    transcription_uploads_dir = Path(str(tmpdir.mkdir("transcription_uploads")))
    import uuid
    corpus_uuid = uuid.uuid1()

    corpus_test_dir = corpus_base_dir / str(corpus_uuid)
    mock_corpus = MagicMock(spec_set=["id","name","training", "testing", "validation"])
    mock_corpus.id = 1
    mock_corpus.name = "mock corpus"
    mock_corpus.training.utterance.__iter__.return_value = ['0']
    mock_corpus.testing.utterance.__iter__.return_value = ['1']
    mock_corpus.validation.utterance.__iter__.return_value = ['2']

    create_corpus_file_structure(audio_uploads_dir, transcription_uploads_dir, mock_corpus, corpus_test_dir)

    train_prefixes_path = corpus_test_dir / "train_prefixes.txt"
    test_prefixes_path = corpus_test_dir / "test_prefixes.txt"
    valid_prefixes_path = corpus_test_dir / "valid_prefixes.txt"

    label_path = corpus_test_dir / "label"
    wav_path = corpus_test_dir / "wav"

    assert corpus_test_dir.is_dir()
    assert train_prefixes_path.is_file()
    assert test_prefixes_path.is_file()
    assert valid_prefixes_path.is_file()

    assert label_path.is_dir()
    assert wav_path.is_dir()
