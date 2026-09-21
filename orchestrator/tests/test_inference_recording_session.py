from orchestrator.internal.inference_recording import InferenceRecordingSession


def test_session_retained_until_identity_changes(tmp_path):
    session = InferenceRecordingSession(tmp_path)
    session.configure('robot', 'act', '/models/a')
    saved = session.prepare('robot')
    session.configure('robot', 'act', '/models/a/')
    assert session.prepare('robot') == saved
    session.configure('robot', 'act', '/models/b')
    assert session.session_id == ''
    session.select(saved, 'robot')
    assert session.prepare('robot') == saved
    session.configure('other', 'act', '/models/b')
    assert session.session_id == ''


def test_new_selection_and_restart_do_not_delete_previous_folder(tmp_path):
    session = InferenceRecordingSession(tmp_path)
    old = session.prepare('robot')
    session.select('', 'robot')
    assert session.session_id == ''
    new = session.prepare('robot')
    assert old != new
    assert len(list(tmp_path.iterdir())) == 2
    assert InferenceRecordingSession(tmp_path).session_id == ''
