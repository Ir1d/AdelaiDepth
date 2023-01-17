import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from lib.test_utils import refine_focal, refine_focal_steps, refine_shift, refine_shift_steps
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
from lib.spvcnn_classsification import SPVCNN_CLASSIFICATION
from lib.test_utils import reconstruct_depth, reconstruct_depthfield

import flask
import requests

import subprocess
from pprint import pprint
import struct
from utils import *

import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    # parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--load_ckpt', default='./res101.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

    args = parser.parse_args()
    return args

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def make_shift_focallength_models():
    shift_model = SPVCNN_CLASSIFICATION(input_channel=3,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    focal_model = SPVCNN_CLASSIFICATION(input_channel=5,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    shift_model.eval()
    focal_model.eval()
    return shift_model, focal_model

def reconstruct3D_from_depth(rgb, pred_depth, shift_model, focal_model, fov):
    cam_u0 = rgb.shape[1] / 2.0
    cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5

    dmax = np.percentile(pred_depth_norm, 100)
    pred_depth_norm = pred_depth_norm / dmax

    originalFov = fov





    # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
    # fov2 = 60
    fov2 = fov
    proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((fov2/2.0)*np.pi/180))

    # recover focal
    focal_scale_1 = refine_focal(pred_depth_norm, proposed_scaled_focal, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()
    # predicted_focal_1 = proposed_scaled_focal
    predictedFov = 2 * np.arctan(rgb.shape[0] / (2 * predicted_focal_1)) * 180 / np.pi

    # fov = predictedFov

    # recover shift
    steps = 1
    fov3 = fov
    # fov3 = 60
    predicted_focal_3 = (rgb.shape[0] // 2 / np.tan((fov3/2.0)*np.pi/180))
    shift_1 = refine_shift_steps(pred_depth_norm, shift_model, predicted_focal_3, cam_u0, cam_v0, steps)
    # shift_1 = shift_1 if shift_1.item() < 0.6 else torch.tensor([0.6])
    # factor = fov2 / predicted_fov_x
    # pred_depth_norm = pred_depth_norm * factor
    depth_scale_1 = pred_depth_norm
    # depth_scale_1 = pred_depth_norm - shift_1.item() * 0.5
    # if shift_1.item() > 0:
      # depth_scale_1 = pred_depth_norm - shift_1.item() * 0.25
    # else:
      # depth_scale_1 = pred_depth_norm

    multiplier = 1
    if shift_1.item() >= 0:
      multiplier = 2.25
    else:
      multiplier = 0
    focalLengthFactor = 1 - shift_1.item() * multiplier

    depth_scale_1 = depth_scale_1 * focalLengthFactor





    print('fov: ', fov)
    print('original fov: ', originalFov)
    print('predicted fov: ', predictedFov)
    print('focal: ', predicted_focal_1)
    print('shift: ', shift_1.item())
    print('focalLengthFactor:', focalLengthFactor)

    fl = (rgb.shape[0] // 2 / np.tan((fov/2.0)*np.pi/180))
    return shift_1, fl, depth_scale_1, fov

def getFov(rgb, pred_depth, focal_model, fov, steps):
    cam_u0 = rgb.shape[1] / 2.0
    cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5

    dmax = np.percentile(pred_depth_norm, 99)
    pred_depth_norm = pred_depth_norm / dmax

    # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
    proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((fov/2.0)*np.pi/180))

    # recover focal
    focal_scale_1 = refine_focal_steps(pred_depth_norm, proposed_scaled_focal, focal_model, u0=cam_u0, v0=cam_v0, steps=steps)
    predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()
    predicted_focal_2 = predicted_focal_1
    fov = 2 * np.arctan(rgb.shape[0] / (2 * predicted_focal_2)) * 180 / np.pi
    return fov


args = parse_args()

# create depth model
depth_model = RelDepthModel(backbone=args.backbone)
depth_model.eval()

# create shift and focal length model
shift_model, focal_model = make_shift_focallength_models()

# load checkpoint
load_ckpt(args, depth_model, shift_model, focal_model)
depth_model.cuda()
shift_model.cuda()
focal_model.cuda()

# flask server
app = flask.Flask(__name__)

# serve api routes
@app.route("/pointcloud", methods=["POST", "OPTIONS"])
def predict():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    # image_dir = os.path.dirname(os.path.dirname(__file__)) + '/test_images/'
    # imgs_list = os.listdir(image_dir)
    # imgs_list.sort()
    # imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
    # image_dir_out = image_dir + '/outputs'
    # os.makedirs(image_dir_out, exist_ok=True)

    # for i, v in enumerate(imgs_path):

    # get body bytes
    body = flask.request.get_data()

    # print('processing (%04d)-th image... %s' % (i, v))
    # rgb = cv2.imread(v)
    rgb = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    rgb_c = rgb[:, :, ::-1].copy()
    gt_depth = None
    A_resize = cv2.resize(rgb_c, (448, 448))
    rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

    img_torch = scale_torch(A_resize)[None, :, :, :]
    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

    # check if we have the forceFov argument
    fov = 60
    if 'forceFov' in flask.request.args:
        fov = float(flask.request.args['forceFov'])
    else:
        proxyRequest = requests.post("http://127.0.0.1:5555/predictFov", data=body)
        if proxyRequest.status_code != 200:
            # proxt the response content back to the client
            response = flask.Response(proxyRequest.content)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Expose-Headers"] = "*"
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
            response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
            return response
            
        json = proxyRequest.json()
        # get the "focalLength", "fov" and "distortion" floats from the response json
        fov = float(json['fov'])
        # focal_length = float(json['focalLength'])
        # distortion = float(json['distortion'])

    # recover focal length, shift, and scale-invariant depth
    shift, focal_length, depth_scaleinv, fov2 = reconstruct3D_from_depth(rgb, pred_depth_ori, shift_model, focal_model, fov)
    # disp = 1 / depth_scaleinv
    # disp = (disp / disp.max() * 60000).astype(np.uint16)

    # print(f"got fov 2 {fov2}")

    # if GT depth is available, uncomment the following part to recover the metric depth
    #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

    # img_name = v.split('/')[-1]
    # img_name = "image.png"
    # cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
    # # save depth
    # plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
    # np.save(os.path.join(image_dir_out, img_name[:-4]+'-depth.npy'), pred_depth_ori)
    # print(focal_length)

    # cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
    # # save disp
    # cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'.png'), disp)



    rgb2 = np.squeeze(rgb[:, :, ::-1])
    depth2 = np.squeeze(depth_scaleinv)

    mask = depth2 < 1e-8
    depth2[mask] = 0
    depth2 = depth2 / depth2.max() * 10000

    # reconstruct point cloud from the depth
    result_ndarray = reconstruct_depth(depth2, rgb2, focal=focal_length)
    # serialize the ndrarray to the result
    result_bytes = result_ndarray.tobytes()
    # return the result
    response = flask.Response(result_bytes)
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["X-Fov"] = str(fov2)
    return response

# serve api routes
@app.route("/depthfield", methods=["POST", "OPTIONS"])
def depthfield():
    if (flask.request.method == "OPTIONS"):
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response

    # get body bytes
    body = flask.request.get_data()

    rgb = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    rgb_c = rgb[:, :, ::-1].copy()
    gt_depth = None
    A_resize = cv2.resize(rgb_c, (448, 448))
    rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

    img_torch = scale_torch(A_resize)[None, :, :, :]
    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

    # check if we have the forceFov argument
    fov = 60
    if 'forceFov' in flask.request.args:
        fov = float(flask.request.args['forceFov'])
    else:
        proxyRequest = requests.post("http://127.0.0.1:5555/predictFov", data=body)
        if proxyRequest.status_code != 200:
            # proxt the response content back to the client
            response = flask.Response(proxyRequest.content)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Expose-Headers"] = "*"
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
            response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
            return response
            
        json = proxyRequest.json()
        # get the "focalLength", "fov" and "distortion" floats from the response json
        fov = float(json['fov'])
        # focal_length = float(json['focalLength'])
        # distortion = float(json['distortion'])

    # recover focal length, shift, and scale-invariant depth
    shift, focal_length, depth_scaleinv, fov2 = reconstruct3D_from_depth(rgb, pred_depth_ori, shift_model, focal_model, fov)

    rgb2 = np.squeeze(rgb[:, :, ::-1])
    depth2 = np.squeeze(depth_scaleinv)

    mask = depth2 < 1e-8
    depth2[mask] = 0
    depth2 = depth2 / depth2.max() * 10000

    result_bytes = reconstruct_depthfield(depth2)
    # return the result
    response = flask.Response(result_bytes)
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["X-Fov"] = str(fov2)
    # compute focal length from fov:
    # fl = (rgb.shape[0] // 2 / np.tan((fov/2.0)*np.pi/180))
    # response.headers["X-Focal-Length"] = str(focal_length)
    return response

# serve api routes
@app.route("/depth", methods=["POST", "OPTIONS"])
def getDepth():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    # get body bytes
    body = flask.request.get_data()

    proxyRequest = requests.post("http://127.0.0.1:4444/depth", data=body)
    
    # proxy the response content back to the client
    response = flask.Response(proxyRequest.content)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

# API call that returns depth map given post request with image
@app.route("/predictDepth", methods=["POST", "OPTIONS"])
def predictDepth():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    # get body bytes
    try:
        body = flask.request.get_data()

        # print('processing (%04d)-th image... %s' % (i, v))
        # rgb = cv2.imread(v)
        rgb = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        rgb_c = rgb[:, :, ::-1].copy()
        A_resize = cv2.resize(rgb_c, (448, 448))

        img_torch = scale_torch(A_resize)[None, :, :, :]
        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        result_bytes = pred_depth_ori.tobytes()
        # return the result
        response = flask.Response(result_bytes)
        response.headers["Content-Type"] = "application/octet-stream"
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response
    except Exception as e:
        # respond with error message e with 500 response code
        response =  flask.Response(str(e), status=500)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response

        

@app.route("/fov", methods=["POST", "OPTIONS"])
def fov1():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    fov = float(flask.request.args.get("fov", 60.0))
    steps = int(flask.request.args.get("steps", 8))

    # get body bytes
    body = flask.request.get_data()

    # print('processing (%04d)-th image... %s' % (i, v))
    # rgb = cv2.imread(v)
    rgb = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    rgb_c = rgb[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (448, 448))

    img_torch = scale_torch(A_resize)[None, :, :, :]
    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

    # recover focal length, shift, and scale-invariant depth
    fov2 = getFov(rgb, pred_depth_ori, focal_model, fov, steps)

    # convert the ndarray into json for the response
    fov_json = json.dumps({
        "fov": fov2
    })

    # respond with the data
    response = flask.Response(fov_json, mimetype='application/json')
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

@app.route("/predictFov", methods=["POST", "OPTIONS"])
def predictFov():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    # get body bytes
    body = flask.request.get_data()

    proxyRequest = requests.post("http://127.0.0.1:5555/predictFov", data=body)
    proxyResponse = proxyRequest.content

    # respond with the data
    response = flask.Response(proxyResponse, status=proxyRequest.status_code, mimetype='application/json')
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

ransacBinPath = "./PlaneFitting/src/PlaneFittingSample"
@app.route("/ransac", methods=["POST", "OPTIONS"])
def ransac():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    # get body bytes
    points = flask.request.get_data()

    # parse threshold arg (float)
    threshold = float(flask.request.args.get("threshold", 0.01))
    # parse init_n arg (int)
    init_n = int(flask.request.args.get("n", 100))
    # parse iter arg (int)
    iter = int(flask.request.args.get("iter", 1000))
    # parse n arg (int)
    n = int(flask.request.args.get("n", 16))

    # convert to ndarray [x3] of points
    points3 = np.frombuffer(points, dtype=np.float32)
    points3 = points3.reshape(-1, 3)

    planes = []
    while len(points3) >= 3 and len(planes) < n:
        [planeEquation, inlierPointIndices] = PlaneRegression(points3, threshold=threshold, init_n=init_n, iter=iter)
        # acc the plane
        planes.append([planeEquation.tolist(), inlierPointIndices])
        # remove the points at the given inlierPointIndices from points3
        points3 = np.delete(points3, inlierPointIndices, axis=0)

    # convert the ndarray into json for the response
    planes_json = json.dumps(planes)

    # respond with the data
    response = flask.Response(planes_json, mimetype='application/json')
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

planeDetectionBinPath = "./RGBDPlaneDetection/build/RGBDPlaneDetection"
@app.route("/planeDetection", methods=["POST", "OPTIONS"])
def planeDetection():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    # get body bytes
    bodyBytes = flask.request.get_data()

    # parse minSupport arg (int), default 40000
    minSupport = int(flask.request.args.get("minSupport", 40000))
    minSupportString = str(minSupport)

    result = subprocess.run([planeDetectionBinPath, minSupportString], input=bodyBytes, stdout=subprocess.PIPE)

    # respond with the data
    response = flask.Response(result.stdout, mimetype='application/octet-stream')
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

# listen as a threaded server on 0.0.0.0:80
app.run(host="0.0.0.0", port=80, threaded=True)