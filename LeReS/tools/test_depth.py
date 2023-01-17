import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import io
import flask

from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
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

args = parse_args()

# create depth model
depth_model = RelDepthModel(backbone=args.backbone)
depth_model.eval()

# load checkpoint
load_ckpt(args, depth_model, None, None)
depth_model.cuda()

# image_dir = os.path.dirname(os.path.dirname(__file__)) + '/test_images/'
# imgs_list = os.listdir(image_dir)
# imgs_list.sort()
# imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
# image_dir_out = image_dir + '/outputs'
# os.makedirs(image_dir_out, exist_ok=True)

# flask server
app = flask.Flask(__name__)

# serve api route
@app.route("/depth", methods=["POST", "OPTIONS"])
def predict():
    if (flask.request.method == "OPTIONS"):
        print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        print("got options 2")
        return response

    print(f"got regular request {flask.request.url}")
    # the body as binary bytesio
    body = flask.request.get_data()
    # decode to cv2
    rgb = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    # print the width and height
    print(f"got image {rgb.shape[1]}x{rgb.shape[0]}")
    rgb_c = rgb[:, :, ::-1].copy()
    gt_depth = None
    A_resize = cv2.resize(rgb_c, (448, 448))
    rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

    img_torch = scale_torch(A_resize)[None, :, :, :]
    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

    # if GT depth is available, uncomment the following part to recover the metric depth
    #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

    # respond with the image
    # output = cv2.imencode(".png", pred_depth_ori)[1].tobytes()
    # plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
    
    # get the "mode" uql query parameter
    mode = flask.request.args.get("mode")

    # save the image, same as above, with rainbow map, except to a bytesio for the response
    bs = None
    if mode != "rainbow": # monochrome
        # encode with cv2 to bytesio
        bs = cv2.imencode(".png", (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))[1].tobytes()
    else:
        bsio = io.BytesIO()
        plt.imsave(bsio, pred_depth_ori, cmap='rainbow')
        bs = bsio.getvalue()
    
    # make a response with the image
    response = flask.Response(bs, mimetype="image/png")
    # set cors/coop headers
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    # return the response
    return response

    # img_name = v.split('/')[-1]
    # cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
    # # save depth
    # plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
    # cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))

# listen as a threaded server on 0.0.0.0:80
app.run(host="0.0.0.0", port=80, threaded=True)