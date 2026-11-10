import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from mesh_refinement import evaluate_mesh,refine_mesh
from mesh_refinement.batch import run_dataset,read_manifest,aggregate_records,aggregate_runs


def dataset(root,ids=('a','b'),offset=7):
    root.mkdir()
    v=np.array([[0.,0,0],[0.,0,0],[1.,0,0],[0.,1,0]])
    f=np.array([[0,2,3],[1,2,3]])
    gt=np.tile([.4,.6],(4,1));skin=gt.copy();skin[1]=[.8,.2]
    bind=np.tile(np.eye(4),(2,1,1));bind[1,0,3]=1
    for key in ids:
        np.savez(root/(key+'.npz'),vertices=v,faces=f,gt_skin=gt,skin=skin,
            matrix_local=bind,parents=np.array([-1,0]),
            metadata_json=np.asarray(json.dumps(dict(sample_id=key,weights='predicted'))))
    manifest=dict(dataset='fixture',weights='predicted',records=[
        dict(sample_id=key,file=key+'.npz',source_index=offset+3*i) for i,key in enumerate(ids)])
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root/'manifest.json'


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_batch_matches_single_and_preserves_ordinals(self):
        manifest=dataset(self.root/'inputs')
        result=run_dataset(manifest,self.root/'run',pose_count=3)
        record=json.loads((self.root/'run/records/b.json').read_text())
        self.assertEqual(record['sample_index'],10)
        with np.load(self.root/'inputs/b.npz') as data:
            refined=refine_mesh(data['vertices'],data['faces'],data['skin'])['evaluation']
            expected=evaluate_mesh(data['vertices'],data['faces'],data['gt_skin'],refined,
                skeleton={k:data[k] for k in ('matrix_local','parents')},pose_count=3,sample_index=10)
        self.assertEqual(record['methods']['refined']['de'],expected['de'])
        self.assertEqual(result['mesh_count'],2)
        self.assertTrue((self.root/'run/metrics.csv').is_file())
        self.assertEqual(json.loads((self.root/'run/complete.json').read_text())['status'],'complete')
        with self.assertRaises(FileExistsError):run_dataset(manifest,self.root/'run')

    def test_worker_parity(self):
        manifest=dataset(self.root/'inputs')
        a=run_dataset(manifest,self.root/'one',pose_count=2)
        b=run_dataset(manifest,self.root/'two',pose_count=2,workers=2)
        self.assertEqual(a,b)

    def test_manifest_override_gzip_and_rejections(self):
        manifest=dataset(self.root/'inputs')
        doc=json.loads(manifest.read_text());doc['records'][1]['sample_index']=81
        compressed=self.root/'selection.json.gz'
        with gzip.open(compressed,'wt') as stream:json.dump(doc,stream)
        _,rows=read_manifest(compressed,self.root/'inputs')
        self.assertEqual(rows[1]['sample_index'],81)
        for change in ('duplicate','traversal','missing_index'):
            bad=json.loads(json.dumps(doc))
            if change=='duplicate':bad['records'].append(bad['records'][0])
            elif change=='traversal':bad['records'][0]['file']='../outside.npz'
            else:del bad['records'][0]['source_index']
            target=self.root/(change+'.json');target.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):read_manifest(target,self.root/'inputs')

    def test_failure_is_not_complete_or_silently_skipped(self):
        manifest=dataset(self.root/'inputs',ids=('a',))
        np.savez(self.root/'inputs/a.npz',vertices=np.ones((3,3)))
        with self.assertRaises(ValueError):run_dataset(manifest,self.root/'run',pose_count=2)
        self.assertTrue((self.root/'run/failure.json').is_file())
        self.assertFalse((self.root/'run/complete.json').exists())
        self.assertFalse((self.root/'run/summary.json').exists())

    def test_combine_runs_and_detect_tamper_overlap(self):
        a=dataset(self.root/'a',ids=('a',));b=dataset(self.root/'b',ids=('b',))
        run_dataset(a,self.root/'ra',pose_count=2);run_dataset(b,self.root/'rb',pose_count=2)
        combined=aggregate_runs([self.root/'ra',self.root/'rb'],self.root/'combined.json')
        self.assertEqual(combined['mesh_count'],2)
        with self.assertRaises(ValueError):aggregate_runs([self.root/'ra',self.root/'ra'],self.root/'overlap.json')
        (self.root/'ra/records/a.json').write_text('{}')
        with self.assertRaises(ValueError):aggregate_runs([self.root/'ra'],self.root/'tamper.json')

    def test_aggregation_denominators(self):
        template=dict(skin_l1=1.,de=None,fpr=1.,cd=0.,nme=0,so=0.,ofpr_source_pair_count=2,
            ofpr_preserved_pair_count=1,local_skin_l1=2.,local_skin_valid_vertex_count=1,
            local_de=None,local_de_valid_vertex_count=None)
        a=dict(template);b=dict(template,skin_l1=3.,ofpr_source_pair_count=8,
            ofpr_preserved_pair_count=8,local_skin_l1=4.,local_skin_valid_vertex_count=3)
        scores=aggregate_records([dict(methods={name:v for name in ('original','refined')}) for v in (a,b)])['methods']['refined']
        self.assertEqual(scores['skin_l1']['mean'],2)
        self.assertEqual(scores['ofpr']['mean'],.9)
        self.assertEqual(scores['local_skin_l1']['mean'],3.5)
        self.assertIsNone(scores['de']['mean'])

    def test_recorded_paper_manifests(self):
        folder=Path(__file__).resolve().parents[1]/'config/evaluation'
        for split,count in (('axl2_train',46638),('axl2_test',1997),('rignetv1',2703)):
            with gzip.open(folder/(split+'.json.gz'),'rt') as stream:document=json.load(stream)
            self.assertEqual(document['population_size'],count)
            self.assertEqual(len(document['records']),count)
            self.assertEqual(len({r['sample_id'] for r in document['records']}),count)
            self.assertTrue(all(isinstance(r['sample_index'],int) and r['sample_index']>=0
                                and r['file']==r['sample_id']+'.npz' for r in document['records']))


try:
    from PIL import Image
    HAS_PIL=True
except ImportError:
    HAS_PIL=False


@unittest.skipUnless(HAS_PIL,'install .[visualization] for renderer tests')
class VisualizeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);dataset(self.root/'inputs',ids=('a',))
        self.input=self.root/'inputs/a.npz'
        with np.load(self.input) as a:
            refined=refine_mesh(a['vertices'],a['faces'],a['skin'])['evaluation']
        self.refined=self.root/'refined.npz'
        np.savez(self.refined,**{k:v for k,v in refined.items() if isinstance(v,np.ndarray)})

    def test_render_cli_and_motion(self):
        subprocess.run([sys.executable,'-m','mesh_refinement','render','--input',str(self.input),
            '--refined',str(self.refined),'--output-dir',str(self.root/'render'),'--frames','4'],
            check=True,capture_output=True)
        report=json.loads((self.root/'render/render.json').read_text())
        self.assertTrue(report['backface_culling']);self.assertTrue(report['no_inference'])
        with Image.open(self.root/'render/comparison.png') as image:
            self.assertEqual(image.size,(1280,640))
        from mesh_refinement.visualize import render_comparison
        render_comparison(self.input,self.refined,self.root/'repeat',motion=self.root/'render/motion.npz')
        with np.load(self.root/'repeat/motion.npz') as second,np.load(self.root/'render/motion.npz') as first:
            np.testing.assert_array_equal(first['rotvec'],second['rotvec'])
            np.testing.assert_array_equal(first['bounds'],second['bounds'])
        with self.assertRaises(FileExistsError):render_comparison(self.input,self.refined,self.root/'render',frames=4)

    def test_culling(self):
        from mesh_refinement.visualize import render_panel,_camera
        camera=_camera('front');points=np.array([[-.5,-.5,0],[.5,-.5,0],[0,.5,0]])
        points=points@camera.T;bounds=np.array([[-1.,-1.],[1.,1.]])
        front=np.asarray(render_panel(points,np.array([[0,1,2]]),bounds,(100,100),camera))
        back=np.asarray(render_panel(points,np.array([[0,2,1]]),bounds,(100,100),camera))
        self.assertTrue(np.any(front!=250));self.assertTrue(np.all(back==250))

    def test_invalid_motion_and_rest_only(self):
        from mesh_refinement.visualize import render_comparison
        np.savez(self.root/'bad.npz',rotvec=np.zeros((4,3,3)))
        with self.assertRaises(ValueError):render_comparison(self.input,self.refined,self.root/'bad',motion=self.root/'bad.npz')
        with np.load(self.input) as a:
            np.savez(self.root/'rest.npz',**{k:a[k] for k in ('vertices','faces','skin')})
        r=render_comparison(self.root/'rest.npz',self.refined,self.root/'static',frames=3)
        self.assertIn('Static',r['motion'])


if __name__=='__main__':unittest.main()
