"""Side-by-side CPU previews of supplied weights, without model inference."""

from pathlib import Path

import numpy as np

from . import _metric_kernels as K
from .batch import file_hash, write_json
from .core import validate_geometry, validate_skin
from .metrics import _validated_skeleton


def _camera(view):
    if view == 'side':
        return np.stack(([0.,1.,0.],[.6,0.,.8],[.8,0.,-.6]),axis=1)
    if view == 'front':
        return np.stack(([1.,0.,0.],[0.,.3,np.sqrt(.91)],[0.,-np.sqrt(.91),.3]),axis=1)
    raise ValueError('view must be front or side')


def render_panel(vertices, faces, bounds, size, camera):
    from PIL import Image, ImageDraw
    q = vertices @ camera
    width,height = size
    image = Image.new('RGB',(width*2,height*2),(250,250,250))
    draw = ImageDraw.Draw(image)
    scale = min((width*2-28)/(bounds[1,0]-bounds[0,0]),
                (height*2-28)/(bounds[1,1]-bounds[0,1]))
    xy = np.rint((q[:,:2]-bounds.mean(axis=0))*[scale,-scale]+[width,height]).astype(int)
    tri = q[faces]
    normals = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    light=np.array([-.3,.5,.8]);light/=np.linalg.norm(light)
    colors=np.rint((.62+.36*np.abs(normals@light))[:,None]*[44,158,176]).astype(np.uint8)
    for index in np.argsort(tri[:,:,2].mean(axis=1)):
        if normals[index,2] <= 1e-12: continue
        polygon=[tuple(p) for p in xy[faces[index]]]
        draw.polygon(polygon,fill=tuple(int(x) for x in colors[index]))
        draw.line(polygon+[polygon[0]],fill=tuple(int(x) for x in np.rint(colors[index]*.7)),width=1)
    return image.resize(size,Image.Resampling.LANCZOS)


def _label(image,text,position):
    from PIL import Image,ImageDraw,ImageFont
    # Prefer a readable TrueType font when installed; Pillow alone remains sufficient.
    from importlib.util import find_spec
    font_path='DejaVuSans.ttf'
    spec=find_spec('matplotlib')
    if spec is not None:
        font_path=str(Path(next(iter(spec.submodule_search_locations)))/'mpl-data/fonts/ttf/DejaVuSans.ttf')
    try:
        font=ImageFont.truetype(font_path,20);factor=1
    except OSError:
        font=ImageFont.load_default();factor=2
    box=font.getbbox(text)
    label=Image.new('RGB',(box[2]-box[0]+4,box[3]-box[1]+4),(250,250,250))
    ImageDraw.Draw(label).text((2-box[0],2-box[1]),text,font=font,fill=(32,43,52))
    if factor!=1:
        label=label.resize((label.width*factor,label.height*factor),Image.Resampling.LANCZOS)
    image.paste(label,position)


def render_comparison(input_path, refined_path, output_dir, *, view='front',
                      motion=None, frames=48, fps=12, seed=321, angle_deg=30.):
    from PIL import Image,ImageDraw
    from .cli import _load_npz
    output=Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output already exists: '+str(output))
    if isinstance(frames,bool) or not isinstance(frames,int) or frames < 2:
        raise ValueError('frames must be an integer >= 2')
    if isinstance(fps,bool) or not isinstance(fps,int) or not 1 <= fps <= 50:
        raise ValueError('fps must be an integer from 1 to 50')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not np.isfinite(angle_deg) or angle_deg < 0:
        raise ValueError('angle must be finite and nonnegative')
    camera=_camera(view)
    original=_load_npz(input_path,('vertices','faces','skin'))
    refined=_load_npz(refined_path,('vertices','faces','skin','keep_indices'))
    states=[]
    for state in (original,refined):
        v,f=validate_geometry(state['vertices'],state['faces'])
        skin=validate_skin(state['skin'],len(v))
        states.append(dict(vertices=v,faces=f,skin=skin))
    if not len(states[0]['vertices']) or not len(states[1]['vertices']):
        raise ValueError('cannot render an empty mesh')
    joints=states[0]['skin'].shape[1]
    if states[1]['skin'].shape[1] != joints:raise ValueError('joint counts differ')
    keep=np.asarray(refined['keep_indices'])
    if (keep.shape!=(len(states[1]['vertices']),) or keep.dtype.kind not in 'iu' or
            np.any(keep<0) or np.any(keep>=len(states[0]['vertices'])) or
            len(np.unique(keep))!=len(keep) or
            not np.array_equal(states[0]['vertices'][keep],states[1]['vertices'])):
        raise ValueError('refined vertices must match retained original coordinates')
    present=('matrix_local' in original,'parents' in original)
    if any(present) and not all(present):raise ValueError('partial original skeleton')
    skeleton=_validated_skeleton(original,joints) if all(present) else None
    for key in ('matrix_local','parents'):
        if key in refined and (key not in original or not np.array_equal(refined[key],original[key])):
            raise ValueError('original and refined skeletons differ')
    if motion is not None:
        if skeleton is None:raise ValueError('a skeleton is required for supplied motion')
        with np.load(motion,allow_pickle=False) as data: angles=np.asarray(data['rotvec'],dtype=np.float64)
        if angles.ndim!=3 or angles.shape[1:]!=(joints,3) or len(angles)<2 or not np.isfinite(angles).all():
            raise ValueError('motion rotvec must be finite (frames,J,3), in radians')
        description='User-supplied rotation vectors; not resampled or selected by error'
    elif skeleton is not None:
        target=np.random.default_rng(seed).uniform(-np.deg2rad(angle_deg),np.deg2rad(angle_deg),(joints,3))
        target[skeleton['parents']<0]=0
        angles=np.sin(np.arange(frames)*2*np.pi/frames)[:,None,None]*target[None]
        description='Procedural sinusoidal motion; fixed sampled per-joint rotation vectors; roots fixed. Not original asset motion or the metric pose bank.'
    else:
        angles=np.zeros((2,joints,3));description='Static rest-pose comparison; no skeleton supplied'
    if skeleton is not None:
        bases=np.tile(np.eye(4),(len(angles),joints,1,1))
        bases[:,:,:3,:3]=K._axis_angle_to_matrix_batch(angles.reshape(-1,3)).reshape(len(angles),joints,3,3)
        poses=[K._compose_pose_matrix(skeleton['matrix_local'],skeleton['parents'],basis) for basis in bases]
        for state in states:
            state['posed']=np.asarray([K._linear_blend_skinning_np(state['vertices'],skeleton['matrix_local'],pose,state['skin']) for pose in poses])
    else:
        for state in states:state['posed']=np.repeat(state['vertices'][None],len(angles),axis=0)
    projected=[state['posed']@camera for state in states]
    low=np.min([q[:,:,:2].min(axis=(0,1)) for q in projected],axis=0)
    high=np.max([q[:,:,:2].max(axis=(0,1)) for q in projected],axis=0)
    center=(low+high)/2;half=np.maximum((high-low)*.56,1e-6)
    bounds=np.stack((center-half,center+half))
    output.mkdir(parents=True,exist_ok=False)
    images=[]
    for frame in range(len(angles)):
        image=Image.new('RGB',(1280,640),(250,250,250))
        for column,state in enumerate(states):
            offset=column*640
            _label(image,'Original mesh' if column==0 else 'Refined mesh',(offset+150,20))
            _label(image,'{} vertices | {} faces'.format(len(state['vertices']),len(state['faces'])),(offset+100,55))
            image.paste(render_panel(state['posed'][frame],state['faces'],bounds,(640,500),camera),(offset,95))
        ImageDraw.Draw(image).line((640,10,640,600),fill=(220,225,230))
        _label(image,'Supplied skinning weights | Same motion and camera | No inference',(100,610))
        images.append(image)
    images[0].save(output/'comparison.png')
    palette=[image.quantize(colors=128,method=Image.Quantize.MEDIANCUT) for image in images]
    palette[0].save(output/'comparison.gif',save_all=True,append_images=palette[1:],
                    duration=round(1000/fps),loop=0,disposal=2,optimize=True)
    np.savez_compressed(output/'motion.npz',rotvec=angles,camera=camera,bounds=bounds)
    report=dict(motion=description,frames=len(angles),requested_fps=fps,seed=seed,view=view,
        backface_culling=True,wireframe=dict(width_at_2x=1,face_color_scale=.7),
        shared_bounds=True,no_inference=True,input_sha256=file_hash(input_path),
        refined_sha256=file_hash(refined_path),
        files={name:file_hash(output/name) for name in ('comparison.png','comparison.gif','motion.npz')})
    write_json(output/'render.json',report)
    return report
