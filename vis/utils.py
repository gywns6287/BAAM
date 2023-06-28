import numpy as np
import math
import open3d as o3d

def call_scale(vert):
    x = np.max(vert[:,0]) - np.min(vert[:,0])
    y = np.max(vert[:,1]) - np.min(vert[:,1])
    z = np.max(vert[:,2]) - np.min(vert[:,2])
    return np.array([x,y,z])


def euler_angles_to_rotation_matrix(angle, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = angle[0], angle[1], angle[2]

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = np.array(R)

    if is_dir:
        R = R[:, 2]

    return R


def trans_vec_to_mat(rot, trans, dim=4):
    """ project vetices based on extrinsic parameters to 3x4 matrix
    """
    mat = euler_angles_to_rotation_matrix(rot)
    mat = np.hstack([mat, trans.reshape((3, 1))])
    if dim == 4:
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])

    return mat


def project(pose, scale, vertices):
    """ transform the vertices of a 3D car model based on labelled pose
    Input:
        pose: 0-3 rotation, 4-6 translation
        scale: the scale at each axis of the car
        vertices: the vertices position
    """

    if np.ndim(pose) == 1:
        mat = trans_vec_to_mat(pose[:3], pose[3:])
    elif np.ndim(pose) == 2:
        mat = pose

    vertices = vertices * scale
    p_num = vertices.shape[0]

    points = vertices.copy()
    points = np.hstack([points, np.ones((p_num, 1))])
    points = np.matmul(points, mat.transpose())

    return points[:, :3]

class VisOpen3D:

    def __init__(self, width=1920, height=1080, visible=True):
        self.__vis = o3d.visualization.Visualizer()
        self.__vis.create_window(width=width, height=height, visible=visible)
        self.__width = width
        self.__height = height

        if visible:
            self.poll_events()
            self.update_renderer()

    def __del__(self):
        self.__vis.destroy_window()

    def mesh_show_back_face(self):
        self.__vis.get_render_option().mesh_show_back_face = True

    def light_off(self):
        self.__vis.get_render_option().light_on = False

    def light_on(self):
        self.__vis.get_render_option().light_on = True

    def render(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()

    def poll_events(self):
        self.__vis.poll_events()

    def update_renderer(self):
        self.__vis.update_renderer()

    def run(self):
        self.__vis.run()

    def create_window(self,visible):
        self.__vis.create_window(width=self.__width, height=self.__height, visible=visible)

    def destroy_window(self):
        self.__vis.destroy_window()

    def add_geometry(self, data):
        self.__vis.add_geometry(data)

    def update_view_point(self, intrinsic, extrinsic):
        ctr = self.__vis.get_view_control()
        param = self.convert_to_open3d_param(intrinsic, extrinsic)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        self.__vis.update_renderer()

    def get_view_point_intrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        return intrinsic

    def get_view_point_extrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic
        return extrinsic

    def get_view_control(self):
        return self.__vis.get_view_control()

    def save_view_point(self, filename):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, param)

    def load_view_point(self, filename):
        param = o3d.io.read_pinhole_camera_parameters(filename)
        intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic
        self.update_view_point(intrinsic, extrinsic)

    def to_apollo_plane(self):
        intrinsic = np.array([
            [2304.54786556982, 0, 1686.23787612802],
            [0, 2305.875668062, 1354.98486439791],
            [0,0,1]
        ])
        extrinsic = np.identity(n=4)
        self.update_view_point(intrinsic, extrinsic)

    def convert_to_open3d_param(self, intrinsic, extrinsic):
        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = extrinsic
        return param

    def capture_screen_float_buffer(self, show=False):
        image = self.__vis.capture_screen_float_buffer(do_render=True)

        if show:
            plt.imshow(image)
            plt.show()

        return image

    def capture_screen_image(self, filename):
        self.__vis.capture_screen_image(filename, do_render=True)

    def capture_depth_float_buffer(self, show=False):
        depth = self.__vis.capture_depth_float_buffer(do_render=True)

        if show:
            plt.imshow(depth)
            plt.show()

        return depth

    def capture_depth_image(self, filename):
        self.__vis.capture_depth_image(filename, do_render=True)

        # to read the saved depth image file use:
        # depth = o3d.io.read_image(filename)
        # plt.imshow(depth)
        # plt.show()

    def draw_egocar(self, car=0, scale=1, color=[0, 0, 0], rotation=[0, 0, 0], translation=[0, 0, 0]):
        verts, face, _ = load_obj('car_models/0.obj')
        verts = verts.numpy()
        verts[:,[0,1]] *= -1
        verts = project(np.array([-0.2,0,0,0,1.9,0]), [1,1,1], verts)

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(face.verts_idx)
        o3d_mesh.paint_uniform_color(color)
        o3d_mesh.compute_vertex_normals()
        self.add_geometry(o3d_mesh)

    def draw_camera(self, intrinsic, extrinsic, scale=1, color=None):
        # intrinsics
        K = intrinsic

        # convert extrinsics matrix to rotation and translation matrix
        extrinsic = np.linalg.inv(extrinsic)
        R = extrinsic[0:3, 0:3]
        t = extrinsic[0:3, 3]

        width = self.__width
        height = self.__height

        geometries = draw_camera(K, R, t, width, height, scale, color)
        for g in geometries:
            self.add_geometry(g)


    def draw_points3D(self, points3D, color=None):
        geometries = draw_points3D(points3D, color)
        for g in geometries:
            self.add_geometry(g)

    def draw_plane(self, color=[0.5,0.5,0.5], width=100.0, height=0.5, depth=100.0):
        g = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
        gv = np.asarray(g.vertices)
        gv = project(np.array([-0.2,0,0,-width/2,2.6,0]), [1,1,1], gv)
        g.vertices = o3d.utility.Vector3dVector(gv)
        g.paint_uniform_color(color)
        self.add_geometry(g)