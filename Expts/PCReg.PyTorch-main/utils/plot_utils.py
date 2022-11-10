try:
    import open3d as o3d
except:
    print('open3d load fail')
import cv2
import numpy as np

def to_color_map(diff, type=cv2.COLORMAP_JET):
    diff = diff / diff.max() * 255.0
    diff = diff.astype(np.uint8)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)

def visualize_pc(pts, colors=None, size=0.3, window_name='Open3D'):
    # pts (n, 3) numpy
    # colors (n, 3) in RGB
    print(pts.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        if colors.max() > 1:
            colors = colors.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)], window_name=window_name)
    # exit(0)

def visualize_pcd(pcd, size=0.3):
    print(pcd.points)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)])
    # exit(0)