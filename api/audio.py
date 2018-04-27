"""
API endpoints for /audio
This deals with the API access for audio files uploading/downloading.
"""
import flask_uploads
from sqlalchemy import Column, Integer, String

from .upload_config import audio_files
from .orm_base import Base

class Audio(Base):
    """Database ORM definition for Audio files"""
    __tablename__ = 'audio'

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    url = Column(String)

    def __repr__(self):
        return "<Audio(filename={}, url={})>".format(self.filename, self.url)


def post(audioFile):
    """handle POST request for audio file"""
    print("Got {}".format(audioFile))
    try:
        audio_files.save(audioFile)
    except flask_uploads.UploadNotAllowed:
        return "Invalid upload format, must be an audio file", 415
    return "Audio upload not implemented", 501


def get(audioID):
    """Handle GET request for audio file information.
    Note that this does not return the audio file directly but
    rather a JSON object with the relevant information.
    This allows the flexibility of file storage being handled
    by another service that is outside this API service."""
    print("Got audio file ID {}".format(audioID))
    return "Get audio info not implemented", 501
