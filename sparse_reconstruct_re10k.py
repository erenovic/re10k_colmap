import torch
import numpy as np
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms as tf
from jaxtyping import Float, UInt8
from torch import Tensor
from io import BytesIO
from typing import NamedTuple
from einops import rearrange
import math
import argparse
import pycolmap
import open3d as o3d

import struct
import numpy as np
import collections


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


class COLMAPImage(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = COLMAPImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def focal2fov(focal, pixels):
    # I think pixel size is assumed to be 1.0
    return 2*math.atan(pixels/(2*focal))


to_tensor = tf.ToTensor()

def convert_images(
    images: list[UInt8[Tensor, "..."]],
) -> Float[Tensor, "batch 3 height width"]:
    torch_images = []
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes()))
        torch_images.append(to_tensor(image))
    return torch.stack(torch_images)


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    

def retrieve_cam_infos(poses: Float[Tensor, "batch 18"], images: Float[Tensor, "batch 3 height width"], scene_id: str) -> tuple[
    Float[Tensor, "batch 4 4"],  # extrinsics
    Float[Tensor, "batch 3 3"],  # intrinsics
]:
    cam_infos = []

    height, width = images.shape[2:]

    for idx, (pose, image) in enumerate(zip(poses, images)):
        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        fx, fy, cx, cy = pose[:4]
        intrinsics[0, 0] = width * fx
        intrinsics[1, 1] = height * fy
        intrinsics[0, 2] = width * cx
        intrinsics[1, 2] = height * cy

        FovX = focal2fov(fx * width, width)
        FovY = focal2fov(fy * height, height)

        # TODO: I believe the w2c matrix is enough to acquire R and T.
        # Check this assumption!
        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3] = rearrange(pose[6:], "(h w) -> h w", h=3, w=4)
        w2c = w2c.numpy()

        R = w2c[:3, :3]
        T = w2c[:3, 3]

        # Conver PyTorch image to PIL Image. This is required in `cameraList_from_camInfos` during
        # `Scene` class instantiation.
        image = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype(np.uint8))

        # cam_info = CameraInfo(
        #     uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
        #     image_path=scene_id, image_name=f"{idx}", width=width, height=height
        # )
        cam_info = {
            "uid": idx,
            "intrinsics": intrinsics,
            "R": R,
            "T": T,
            "FovY": FovY,
            "FovX": FovX,
            "image": image,
            "image_path": scene_id,
            "image_name": f"{idx}",
            "width": width,
            "height": height
        }
        cam_infos.append(cam_info)

    return cam_infos


def rotmat2qvec(R: np.array) -> np.array:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def load_sample(data_path: str, chosen_sample_key: str):
    train_dir = Path(data_path)
    chunk_id = 0
    chunk = torch.load(train_dir / f'{chunk_id:06d}.torch')

    all_cam_infos = []
    samples = {}

    for sample in chunk:

        if (sample['key'] != chosen_sample_key) and chosen_sample_key != "all":
            continue

        print(sample['key'], sample['url'])
        print(f"Number of images: {len(sample['images'])}")

        images = convert_images(sample['images'])
        cam_infos = retrieve_cam_infos(sample['cameras'], images, sample['key'])

        # print(cam_infos)
        print(images.shape)

        if sample['key'] not in samples:
            samples[sample['key']] = {
                'images': images,
                'cameras': cam_infos
            }
        else:
            print(f"Sample {sample['key']} already exists!")

        os.makedirs(f"{save_dir}/images", exist_ok=True)
        for idx, cam_info in enumerate(cam_infos):
            pil_image = cam_info['image']
            pil_image.save(f"{save_dir}/images/{idx:04d}.png")
            
    return samples


def write_colmap_cameras_and_images(save_dir: str, samples: dict, chosen_sample_key: str):
    os.makedirs(f"{save_dir}/sparse", exist_ok=True)

    if os.path.exists(f"{save_dir}/sparse/cameras.txt"):
        os.remove(f"{save_dir}/sparse/cameras.txt")
        os.system(f"touch {save_dir}/sparse/cameras.txt")
    if os.path.exists(f"{save_dir}/sparse/images.txt"):
        os.remove(f"{save_dir}/sparse/images.txt")
        os.system(f"touch {save_dir}/sparse/images.txt")

    os.system(f"touch {save_dir}/sparse/points3D.txt")

    # Create a template which takes for the lines in cameras.txt
    # Line format similar to: 
    # 1 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
    intr_template = lambda idx, width, height, focal_length, cx, cy: \
        f"{idx} SIMPLE_PINHOLE {width} {height} {focal_length} {cx} {cy}"

    # Create a template for the lines in images.txt
    # Line format similar to:
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    extr_template = lambda idx, qw, qx, qy, qz, tx, ty, tz, cam_id, name: \
        f"{idx} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {name}"

    for key, sample in samples.items():
        
        if (key != chosen_sample_key) and (chosen_sample_key != "all"):
            continue

        print(f"Sample {key} has {len(sample['cameras'])} cameras")
        for cam_info, image in zip(sample['cameras'], sample['images']):
            
            ### Write camera intrinsics to cameras.txt
            focal_length = (
                cam_info['intrinsics'][0, 0].item() + \
                cam_info['intrinsics'][1, 1].item()
            ) / 2.0

            cam_line = intr_template(
                cam_info['uid'] + 1, 
                cam_info['width'], 
                cam_info['height'], 
                f"{focal_length:.2f}", 
                int(cam_info['intrinsics'][0, 2].item()), 
                int(cam_info['intrinsics'][1, 2].item())
            )

            # Write cameras to cameras.txt
            with open(f"{save_dir}/sparse/cameras.txt", 'a') as file:
                file.write(cam_line + '\n')

            ### Write camera extrinsics to images.txt
            # print(cam_info['R'], cam_info['image_name'])
            quat = rotmat2qvec(cam_info['R'])
            trans = cam_info['T']

            extr_line = extr_template(
                cam_info['uid'] + 1, 
                quat[0], quat[1], quat[2], quat[3], 
                trans[0], trans[1], trans[2], 
                cam_info['uid'] + 1, 
                f"{cam_info['uid']:04d}.png"
            )

            # Write extrinsics to images.txt
            with open(f"{save_dir}/sparse/images.txt", 'a') as file:
                file.write(extr_line + '\n')
                file.write('\n')


def update_colmap_database_w_intrinsics(save_dir: str, samples: dict, chosen_sample_key: str):
    database_path = f"{save_dir}/database.db"
    pycolmap_database = pycolmap.Database(database_path)

    for key, sample in samples.items():
        if key != chosen_sample_key:
            continue

        print(f"Sample {key} has {len(sample['cameras'])} cameras")

        for cam_info, image in zip(sample['cameras'], sample['images']):
            
            ### Write camera intrinsics to cameras.txt
            focal_length = (
                cam_info['intrinsics'][0, 0].item() + \
                cam_info['intrinsics'][1, 1].item()
            ) / 2.0

            cam = pycolmap.Camera.create(
                camera_id=cam_info['uid'] + 1,
                model=0,        #SIMPLE_PINHOLE
                focal_length=focal_length,
                width=cam_info['width'],
                height=cam_info['height']
            )
            pycolmap_database.update_camera(cam)

    print(f"Updated cameras in the database at {database_path}!")
    pycolmap_database.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str)     # e.g. "67bd3eefd07e6042"
    parser.add_argument("--data_path", type=str)    # e.g. "../dataset_subsets/re10k_subset/train"

    args = parser.parse_args()

    chosen_sample_key = args.scene_id
    save_dir = f"scene_{chosen_sample_key}"

    if os.path.exists(save_dir):
        os.system(f"rm -rf {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Prepare the sample data for COLMAP
    samples = load_sample(args.data_path, chosen_sample_key)
    write_colmap_cameras_and_images(save_dir, samples, chosen_sample_key)

    # Extract features using COLMAP
    try:
        os.remove(f"{save_dir}/database.db")
        os.remove(f"{save_dir}/database.db-shm")
        os.remove(f"{save_dir}/database.db-wal")
    except:
        print("Database files do not exist!")

    os.system(f"""
    colmap feature_extractor \
        --ImageReader.camera_model SIMPLE_PINHOLE \
        --database_path {save_dir}/database.db \
        --image_path {save_dir}/images
    """
    )

    # Update the database with intrinsics
    update_colmap_database_w_intrinsics(save_dir, samples, chosen_sample_key)

    # Match features using COLMAP
    os.system(f"colmap exhaustive_matcher --database_path {save_dir}/database.db")

    # Perform sparse reconstruction using COLMAP (triangulation)
    os.system(f"""
    colmap point_triangulator \
        --database_path {save_dir}/database.db \
        --image_path {save_dir}/images \
        --input_path {save_dir}/sparse/ \
        --output_path {save_dir}/sparse/"""
    )

    # Check point cloud statistics
    xyzs, rgbs, errors = read_points3D_binary(f"{save_dir}/sparse/points3D.bin")
    print(f"Point cloud binary saved at {save_dir}/sparse/points3D.bin")
    print(f"Number of points: {len(xyzs)}")


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzs)
    pcd.colors = o3d.utility.Vector3dVector(rgbs / 255.0)

    # Save the point cloud to a ply file
    o3d.io.write_point_cloud(f"{save_dir}/points3D.ply", pcd)
    print(f"Point cloud saved at {save_dir}/points3D.ply")

    # Move data to 0 folder
    os.makedirs(f"{save_dir}/sparse/0", exist_ok=True)
    os.system(f"mv {save_dir}/sparse/*.txt {save_dir}/sparse/0/")
    os.system(f"mv {save_dir}/sparse/*.bin {save_dir}/sparse/0/")

    print(f"Data saved at {save_dir}/sparse/0/")
    print("     Data saved in the correct format for COLMAP initialization in Gaussian Splatting!")


    


