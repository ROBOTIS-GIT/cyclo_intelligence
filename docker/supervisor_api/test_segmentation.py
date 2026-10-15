"""Segmentation preview/review tests use synthetic masks and temporary datasets."""
import base64
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

# Load through the same package namespace as app.py (relative imports).
parent = str(Path(__file__).resolve().parent.parent)
if parent not in sys.path: sys.path.insert(0, parent)
from supervisor_api import collection as c
from supervisor_api import segmentation as s
JPEG = b'\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03' + b'\x00' * 10 + b'\xff\xd9'


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.catalog = [{'mission_id':'m1','mission_name':'Mission1','revision':1}]
        self.store = c.CollectionStore(Path(self.temp.name) / 'data',lambda _:self.catalog)
        self.ep = self.store.start(c.EpisodeRequest(scenario='Shelf', group_id='shelf', robot_type='ffw', map_name='warehouse'))
        self.frame = self.store.capture(self.ep['id'], {'image_base64':base64.b64encode(JPEG).decode(),
            'captured_at':1,'frame_stamp':'1','camera_topic':'/head'}, {})
        self.start = Mock()
        self.reply = {'rle':{'size':[2,3],'counts':[2,2,2]}, 'score':.95, 'model':'sam2', 'model_revision':'pinned', 'elapsed_seconds':.1}
        self.request = Mock(side_effect=lambda path,payload=None: {} if path == '/health' else self.reply)
        self.service = s.SegmentationService(self.store, self.start, self.request)
        self.preview_request = s.PreviewRequest(frame_id=self.frame['frame_id'], points=[s.Point(x=1,y=1,label=1)])

    def preview(self):
        return self.service.preview(self.ep['id'], self.preview_request)

    def save(self, preview, **values):
        data = dict(preview_id=preview['preview_id'],frame_id=self.frame['frame_id'],object_id='stick',
                    object_label='wooden_stick',annotator='tester',reviewed=True)
        data.update(values)
        return self.service.save(self.ep['id'], s.SaveRequest(**data))

    def test_point_preview_and_capture_transition_routes_do_not_collide(self):
        app = FastAPI()
        # Match production registration order, so duplicate URLs cannot hide here.
        app.include_router(c.create_router(self.store))
        app.include_router(s.create_router(self.service))
        client = TestClient(app)
        prefix = f"/collection/episodes/{self.ep['id']}"
        paths = [(route.path, method) for route in app.routes
                 for method in getattr(route, 'methods', [])]
        self.assertEqual(len(paths), len(set(paths)))
        response = client.post(prefix + '/segmentation/preview', json=self.preview_request.model_dump())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['frame_id'], self.frame['frame_id'])
        self.start.assert_called_once()  # Normal clicks start SAM on demand.
        self.assertEqual(self.service.list_regions(self.ep['id']), [])
        self.assertEqual(len(self.store.list()), 1)
        with patch.object(self.store, 'start_segment', return_value={'id':'next'}) as transition:
            response = client.post(prefix + '/segment', json={'map_name':'warehouse'})
            self.assertEqual(response.status_code, 200, response.text)
            transition.assert_called_once()

    def test_click_preview_does_not_save_until_review(self):
        preview = self.preview()
        self.assertEqual(self.service.list_regions(self.ep['id']), [])
        self.assertFalse(self.store.path('raw',self.ep['id'],'masks').exists())
        saved = self.save(preview)
        self.assertEqual(saved['area'],2)
        self.assertEqual(saved['bbox_xywh'],[1,0,1,2])
        self.assertEqual(saved['image_sha256'],self.frame['image']['sha256'])
        self.assertTrue(self.store.path('raw',self.ep['id'],*saved['mask']['path'].split('/')).read_bytes().startswith(b'\x89PNG'))

    def test_region_mission_uses_current_design_catalog_and_keeps_provenance(self):
        self.catalog.append({'mission_id':'observe','mission_name':'ObservationMode','revision':2})
        saved=self.save(self.preview(),mission_id='observe')
        self.assertEqual(saved['mission_id'],'observe')
        self.assertEqual(saved['catalog_snapshot_id'],c.digest(self.catalog))
        self.catalog=[self.catalog[0]]
        with self.assertRaises(HTTPException) as error:
            self.save(self.preview(),object_id='new',mission_id='observe')
        self.assertIn('current Design catalog',error.exception.detail)

    def test_saved_frame_bytes_and_prompt_coordinates_reach_sam(self):
        self.preview()
        payload=self.request.call_args.args[1]
        self.assertEqual(base64.b64decode(payload['image_base64']),JPEG)
        self.assertEqual(payload['points'],[{'x':1,'y':1,'label':1}])

    def test_out_of_image_and_negative_only_clicks_never_start_service(self):
        for point in (s.Point(x=3,y=0,label=1),s.Point(x=1,y=1,label=0)):
            with self.assertRaises(HTTPException):
                self.service.preview(self.ep['id'],s.PreviewRequest(frame_id=self.frame['frame_id'],points=[point]))
        self.start.assert_not_called()

    def test_invalid_rle_is_rejected(self):
        for rle in ({'size':[2,3],'counts':[3,-1,4]}, {'size':[2,3],'counts':[1]}, {'size':[2,3],'counts':[6]}):
            with self.assertRaises(HTTPException):s.mask_artifacts(rle,3,2)

    def test_wrong_frame_stale_revision_and_expired_preview_fail(self):
        preview=self.preview()
        with self.assertRaises(HTTPException):self.save(preview,frame_id='wrong')
        self.save(preview)
        with self.assertRaises(HTTPException):self.save(preview)
        updated=self.save(preview,expected_revision=1)
        self.assertEqual(updated['revision'],2)
        self.service.previews[preview['preview_id']]['expires']=0
        with self.assertRaises(HTTPException):self.save(preview,expected_revision=2)

    def test_disk_failure_does_not_append_approved_region(self):
        preview=self.preview()
        with patch.object(self.store,'save_samples',side_effect=OSError('disk full')):
            with self.assertRaises(HTTPException) as caught:self.save(preview)
        self.assertEqual(caught.exception.status_code,507)
        self.assertEqual(self.service.list_regions(self.ep['id']),[])
        self.assertFalse(self.store.path('raw',self.ep['id'],'masks','stick','1.png').exists())

    def test_mask_only_export_and_rejection(self):
        saved=self.save(self.preview());self.store.end(self.ep['id'])
        export=self.store.export([self.ep['id']])
        self.assertEqual(export['approved_regions'],1)
        root=Path(export['storage_path']);row=self.store.episode(self.ep['id'])['training']['regions'][0]
        self.assertTrue((root/row['image_path']).is_file())
        self.assertTrue((root/row['mask']['path']).is_file())
        self.service.reject(self.ep['id'],'stick',s.RejectRequest(expected_revision=1,annotator='reviewer'))
        with self.assertRaises(HTTPException):self.store.export([self.ep['id']])
        self.assertEqual(len(self.store.records(self.ep['id'], 'regions')),2)

    def test_api_requires_explicit_review_and_never_accepts_mask_from_browser(self):
        app=FastAPI();app.include_router(s.create_router(self.service));client=TestClient(app)
        url=f"/collection/episodes/{self.ep['id']}/regions"
        response=client.post(url,json={'reviewed':False})
        self.assertEqual(response.status_code,422)
        self.assertEqual(client.get(url).json(),{'regions':[]})


    def test_batch_saves_separate_objects_and_renames_without_new_inference(self):
        previews=[self.preview(),self.preview()]
        items=[s.BatchItem(preview_id=p['preview_id'],frame_id=self.frame['frame_id'],object_id=str(i),
            object_label=label,annotator='tester',reviewed=True) for i,(p,label) in enumerate(zip(previews,['stick','box']))]
        batch=s.BatchSaveRequest(frame_id=self.frame['frame_id'],items=items)
        records=self.service.save_batch(self.ep['id'],batch)['regions']
        self.assertEqual([r['object_label'] for r in records],['stick','box'])
        self.assertEqual({r['instance_number'] for r in records},{1,2})
        updated=s.BatchItem(frame_id=self.frame['frame_id'],object_id='0',object_label='red stick',annotator='tester',reviewed=True,expected_revision=1)
        result=self.service.save_batch(self.ep['id'],s.BatchSaveRequest(frame_id=self.frame['frame_id'],items=[updated]))['regions'][0]
        self.assertEqual(result['instance_number'],1)
        self.assertEqual(result['revision'],2)
        self.assertEqual(result['segmentation'],records[0]['segmentation'])

    def test_invalid_batch_has_no_partial_saved_masks(self):
        preview=self.preview()
        items=[s.BatchItem(preview_id=token,frame_id=self.frame['frame_id'],object_id=str(i),object_label='stick',annotator='tester',reviewed=True)
               for i,token in enumerate([preview['preview_id'],'missing'])]
        with self.assertRaises(HTTPException):
            self.service.save_batch(self.ep['id'],s.BatchSaveRequest(frame_id=self.frame['frame_id'],items=items))
        self.assertEqual(self.service.list_regions(self.ep['id']),[])
        self.assertFalse(self.store.path('raw',self.ep['id'],'masks').exists())

    def test_batch_publish_failure_preserves_previous_records(self):
        preview=self.preview()
        item=s.BatchItem(preview_id=preview['preview_id'],frame_id=self.frame['frame_id'],object_id='a',object_label='stick',annotator='tester',reviewed=True)
        with patch.object(self.store,'save_samples',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.service.save_batch(self.ep['id'],s.BatchSaveRequest(frame_id=self.frame['frame_id'],items=[item]))
        self.assertEqual(self.service.list_regions(self.ep['id']),[])
        self.assertFalse(list(self.store.path('raw',self.ep['id'],'masks').rglob('*.png')))

    def test_frame_review_approves_all_classes_and_tracks_all_rejected_frames(self):
        for label in ('person', 'dish', 'empty dish'):
            self.save(self.preview(), object_id=label.replace(' ', '-'), object_label=label)
        samples=c.rows(self.store.path('raw',self.ep['id'],'samples.jsonl'))
        for region in samples[0]['regions']:
            region['review_status']='draft'
        self.store.save_samples(self.ep['id'],samples)
        summary=self.store.list()[0]
        self.assertEqual(summary['mask_reviewed_count'],0)
        self.assertEqual(summary['mask_pending_count'],1)
        items=[s.BatchItem(frame_id=self.frame['frame_id'],object_id=label,object_label=label,
            annotator='collect-ui',reviewed=True,expected_revision=1) for label in ('person','dish')]
        self.request.reset_mock()
        with patch.object(self.store,'save_samples',wraps=self.store.save_samples) as publish:
            self.service.save_batch(self.ep['id'],s.BatchSaveRequest(frame_id=self.frame['frame_id'],items=items,
                removed=[s.RemovedItem(object_id='empty-dish',expected_revision=1)]))
        publish.assert_called_once()
        self.request.assert_not_called() # Reviewing existing masks does not run SAM.
        latest=self.service.list_regions(self.ep['id'])
        self.assertEqual({r['object_label']:r['review_status'] for r in latest},
            {'person':'approved','dish':'approved','empty dish':'rejected'})
        summary=self.store.list()[0]
        self.assertEqual(summary['mask_reviewed_count'],1)
        self.assertEqual(summary['mask_labeled_count'],1)
        self.assertEqual(summary['mask_pending_count'],0)
        self.service.save_batch(self.ep['id'],s.BatchSaveRequest(frame_id=self.frame['frame_id'],
            removed=[s.RemovedItem(object_id=label,expected_revision=2) for label in ('person','dish')]))
        summary=self.store.list()[0]
        self.assertEqual(summary['mask_reviewed_count'],1)
        self.assertEqual(summary['mask_labeled_count'],0)
        self.assertEqual(summary['mask_pending_count'],0)

    def test_empty_review_persists_without_masks_and_blocks_auto_predictions(self):
        result=self.service.save_batch(self.ep['id'],s.BatchSaveRequest(frame_id=self.frame['frame_id'],review_empty=True))
        self.assertEqual(result['regions'],[])
        restored=self.store.get(self.ep['id'])
        self.assertEqual(restored['observations'][0]['mask_review']['status'],'empty')
        self.assertEqual(self.store.list()[0]['mask_reviewed_count'],1)
        self.assertEqual(self.service.save_predictions(self.ep['id'],self.frame,'stick',[]),0)
        self.save(self.preview())
        self.assertNotIn('mask_review',self.store.get(self.ep['id'])['observations'][0])
        with self.assertRaises(HTTPException):
            self.service.save_batch(self.ep['id'],s.BatchSaveRequest(frame_id=self.frame['frame_id'],review_empty=True))

if __name__=='__main__':unittest.main()
