"""
API endpoints for /model
This deals with the API access for model definitions and metadata
"""
import os
from pathlib import Path
import pickle
import uuid

import flask
from persephone import experiment
from persephone import rnn_ctc
from persephone.corpus_reader import CorpusReader

from ..extensions import db
from ..db_models import DBcorpus, TranscriptionModel
from ..serialization import TranscriptionModelSchema

class MyPickler(pickle._Pickler):
    def save(self, obj):
        print('pickling object  {0} of type {1}'.format(obj, type(obj)))
        try:
            pickle._Pickler.save(self, obj)
        except:
            import pdb; pdb.set_trace()
            raise

# attempting workaround to get TF to pickle properly:
import tensorflow as tf
setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

def create_RNN_CTC_model(model: TranscriptionModel, corpus_storage_path: Path,
                         models_storage_path: Path):
    """Create a persephone RNN CTC model

    :model: The database entry contaning the information about the model attempting
            to be created here.
    :corpus_storage_path: The path the corpuses are stored at.
    :models_storage_path: The path the models are stored at.
    """
    model_path = models_storage_path / model.filesystem_path
    exp_dir = experiment.prep_exp_dir(directory=str(model_path))
    corpus_db_entry = model.corpus
    pickled_corpus_path = corpus_storage_path / corpus_db_entry.filesystem_path / "corpus.p"
    with pickled_corpus_path.open('rb') as pickle_file:
        corpus = pickle.load(pickle_file)

    corpus_reader = CorpusReader(corpus)
    persephone_model = rnn_ctc.Model(
        exp_dir,
        corpus_reader,
        num_layers=model.num_layers,
        hidden_size=model.hidden_size,
        beam_width=model.beam_width,
        decoding_merge_repeated=model.decoding_merge_repeated
        )
    model_pickle_path = model_path / "model.p"
    with model_pickle_path.open('wb') as pickle_file:
        p = MyPickler(pickle_file, protocol=4)
        p.dump(persephone_model)
    # TODO: pickle model at this point?

def search():
    """Handle request to search over all models"""
    results = []
    for row in db.session.query(TranscriptionModel):
        serialized = TranscriptionModelSchema().dump(row).data
        results.append(serialized)
    return results, 200

def get(modelID):
    """ Get a model by its ID"""
    transcription_model = TranscriptionModel.query.get_or_404(modelID)
    result = TranscriptionModelSchema().dump(transcription_model).data
    return result, 200

def post(modelInfo):
    """Create a new transcription model"""
    current_corpus = DBcorpus.query.get_or_404(modelInfo['corpusID'])

    min_epochs = modelInfo.get('minimumEpochs', 0)
    max_epochs = modelInfo.get('maximumEpochs', None)
    if max_epochs and min_epochs > max_epochs:
        return "minimum number of epochs must be smaller than maximum", 400

    early_stopping_steps = modelInfo.get('earlyStoppingSteps', None)
    num_layers = modelInfo.get('numberLayers', 3)
    hidden_size = modelInfo.get('hiddenSize', 250)
    beam_width = modelInfo.get('beamWidth', 100)
    decoding_merge_repeated = modelInfo.get('decodingMergeRepeated', True)

    model_uuid = uuid.uuid1()

    current_model = TranscriptionModel(
        name=modelInfo['name'],
        corpus=current_corpus,
        num_layers=num_layers,
        hidden_size=hidden_size,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        beam_width=beam_width,
        decoding_merge_repeated=decoding_merge_repeated,
        early_stopping_steps=early_stopping_steps,
        filesystem_path=str(model_uuid)
    )

    db.session.add(current_model)

    create_RNN_CTC_model(
        current_model,
        corpus_storage_path=Path(flask.current_app.config['CORPUS_PATH']),
        models_storage_path=Path(flask.current_app.config['MODELS_PATH'])
    )
    try:
        db.session.commit()
    except sqlalchemy.exc.IntegrityError:
        return "Invalid model provided", 400
    else:
        result = TranscriptionModelSchema().dump(current_model).data
        return result, 201
