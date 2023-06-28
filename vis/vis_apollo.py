import argparse, os, json
import copy

parser = argparse.ArgumentParser()
parser.add_argument('-output', '--output')
parser.add_argument('-file', '--file')
parser.add_argument('-save', '--save')
args = parser.parse_args()

import utils as uts
from pytorch3d.io import load_obj
import numpy as np
import open3d as o3d

# setting
os.makedirs(args.save,exist_ok=True)
# load vert index
_, face, _ = load_obj('../apollo_deform/0.obj')
face = face.verts_idx.numpy()
# load class scales
scales = {}
for i in range(79):
    vert, _, _ = load_obj('car_models/{}.obj'.format(i))
    vert = vert.numpy()
    scales[i] = uts.call_scale(vert)
# load perdiction results
dts = json.load(
    open(os.path.join(args.output,'res',args.file+'.json'))
)
meshes = np.load(
    os.path.join(args.output,'mesh',args.file+'.npy')
)
N = len(dts)
colors = np.random.random((N, 3)) * 255

# load meshes as open3D format
o3d_list = []
vis_thr = 0.1
for idx, (dt, mesh) in enumerate(zip(dts, meshes)):
    if dt['score'] < vis_thr: continue
    # re-scaling
    mesh[:,[0,1]] *= -1
    s = scales[dt['car_id']]
    ts = uts.call_scale(mesh)
    s = s/ts
    # project poses
    mesh = uts.project(np.array(dt['pose']), s, mesh)
    # append open3d mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(copy.deepcopy(mesh))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(face)
    o3d_mesh.paint_uniform_color(colors[idx]/255)
    o3d_mesh.compute_vertex_normals()
    o3d_list.append(o3d_mesh)

# save render image
vis = uts.VisOpen3D(width=3384, height=2710,visible=False)
for obj in o3d_list:
    vis.add_geometry(obj)
vis.mesh_show_back_face()
vis.to_apollo_plane()
vis.capture_screen_image(os.path.join(args.save, f'{args.file}.image_plane.png'))
del vis

# vis 3D field to open3D UI
vis = uts.VisOpen3D(width=1920, height=1080,visible=True)
for obj in o3d_list:
    vis.add_geometry(obj)
vis.mesh_show_back_face()
vis.load_view_point("view_point_3d.json")
vis.run()
vis.capture_screen_image(os.path.join(args.save, f'{args.file}.3d.png'))
