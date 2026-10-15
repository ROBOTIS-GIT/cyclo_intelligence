"""Regression checks for Cosmos local-only resolution, without GPU/model imports."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

MODULE = Path(__file__).resolve().parents[1] / "runtime/cached_cosmos_processor.py"
spec = importlib.util.spec_from_file_location("cached_cosmos_under_test", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CacheResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
            (self.directory / name).write_text('{}')
        self.constants = types.SimpleNamespace(HF_HUB_OFFLINE=True)
        self.download = Mock(return_value=str(self.directory))
        self.missing = type("LocalEntryNotFoundError", (Exception,), {})
        hub = types.ModuleType('huggingface_hub')
        hub.constants = self.constants
        hub.snapshot_download = self.download
        errors = types.ModuleType('huggingface_hub.errors')
        errors.LocalEntryNotFoundError = self.missing
        self.patch = patch.dict(sys.modules, {'huggingface_hub':hub, 'huggingface_hub.errors':errors})
        self.patch.start(); self.addCleanup(self.patch.stop)

    def test_cached_revision_uses_directory_and_keeps_callers_kwargs(self):
        kwargs = {'revision':'pinned', 'cache_dir':'/custom'}
        source, options = m._local_processor_source(m.COSMOS_MODEL, kwargs)
        self.assertEqual(source,str(self.directory))
        self.assertTrue(options['local_files_only'])
        self.assertNotIn('local_files_only',kwargs)
        self.download.assert_called_once_with(repo_id=m.COSMOS_MODEL,revision='pinned',cache_dir='/custom',local_files_only=True)

    def test_other_models_and_explicit_paths_do_not_resolve_cosmos(self):
        for name in ('/models/cosmos', 'another/model'):
            self.assertEqual(m._local_processor_source(name,{}),(name,{}))
        self.download.assert_not_called()

    def test_missing_or_partial_cache_fails_clearly_offline(self):
        (self.directory/'tokenizer.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'not fully cached'):
            m._local_processor_source(m.COSMOS_MODEL,{})
        self.download.side_effect=self.missing('not cached')
        with self.assertRaisesRegex(RuntimeError,'not fully cached'):
            m._local_processor_source(m.COSMOS_MODEL,{})

    def test_online_cache_miss_preserves_upstream_download(self):
        self.constants.HF_HUB_OFFLINE=False
        self.download.side_effect=self.missing('not cached')
        self.assertEqual(m._local_processor_source(m.COSMOS_MODEL,{}),(m.COSMOS_MODEL,{}))
        with self.assertRaises(RuntimeError):
            m._local_processor_source(m.COSMOS_MODEL,{'local_files_only':True})

    def test_scoped_factory_restored_on_failure(self):
        original=Mock(side_effect=ValueError('processor failed'))
        module=types.SimpleNamespace(build_processor=original)
        with patch.object(m.importlib,'import_module',return_value=module):
            with self.assertRaisesRegex(ValueError,'processor failed'):
                with m.cached_cosmos_processor():
                    module.build_processor(m.COSMOS_MODEL,{})
        self.assertIs(module.build_processor,original)
        original.assert_called_once_with(str(self.directory),{'local_files_only':True})

if __name__=='__main__':unittest.main()
