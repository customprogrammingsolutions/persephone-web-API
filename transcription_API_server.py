import os

import connexion
from connexion.resolver import RestyResolver

from flask import send_from_directory
from flask_uploads import patch_request_class

from api.upload_config import configure_uploads
import api


# Create the API endpoints from YAML specification
app = connexion.FlaskApp(__name__, specification_dir='swagger/')
app.add_api('api_spec.yaml', resolver=RestyResolver('api'))
@app.route('/')
def index():
    return """Access to the API is via the API versioned path prefix
<a href="/{version}">/{version}</a>. The API explorer tool can be found at
<a href="/{version}/ui/">/{version}/ui/</a>, this is the best place to explore the API.
""".format(version="v0.1")

# fetch underlying flask app from the connexion app
flask_app = app.app

# configure the DB
# in-memory sqlite DB for development purposes, will need file backing for persistence
flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from api import db
db.init_app(flask_app)

# create DB tables
with flask_app.app_context():
    db.create_all()

# configure upload paths
flask_app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024 #max 64 MB file upload
flask_app.config['BASE_UPLOAD_DIRECTORY'] = os.path.join(os.getcwd(), 'user_uploads')
configure_uploads(flask_app)

# persephone paths
flask_app.config['CORPUS_PATH'] = os.path.join(os.getcwd(), 'persephone_corpus')


@flask_app.route('/uploads/<path:path>')
def uploaded_file(path):
    """Serve uploaded files for development purposes
    Note this is for development only, serve these files with Apache/nginx in production environments.
    """
    return send_from_directory(flask_app.config['BASE_UPLOAD_DIRECTORY'],
                               path)

app.run(port=8080)

