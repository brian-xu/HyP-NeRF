import torch
import argparse
import json, time, tqdm

from nerf.provider_abo import MetaNeRFDataset
from nerf.network_fcblock import NeRFNetwork,HyPNeRF
from nerf.utils import *

torch.set_grad_enabled(False)

#torch.autograd.set_detect_anomaly(True)

'''
TODO - sanity check after modifying the val code
'''

def get_random_rays(opt,poses,intrinsics,H,W,index=None):
        if index is None:
            index = [0]
        B = len(index) # a list of length 1

        rays = get_rays(poses[index], intrinsics, H, W, -1, patch_size=opt.patch_size)
        results = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'patch_size': opt.patch_size,
        }
               
        results['poses'] = poses
        results['intrinsics'] = intrinsics

        results['num_rays'] = opt.num_rays

        return results

def get_video_rays(poses):
    with open(f"{opt.path}/ABO_rendered/B00BBDF500/metadata.json", 'r') as f:
        transform = json.load(f)
    H = W = 512
    
    frames = transform["views"]
    results = []
    
    # load intrinsics
    if 'fl_x' in transform or 'fl_y' in transform:
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / opt.downscale
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / opt.downscale
    elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
        # blender, assert in radians. already downscaled since we use H/W
        fl_x = W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
        fl_y = H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
        if fl_x is None: fl_x = fl_y
        if fl_y is None: fl_y = fl_x
    else:
        fl_x = fl_y = 443.40496826171875 # focal length for ABO dataset

    cx = (transform['cx'] / opt.downscale) if 'cx' in transform else (W / 2)
    cy = (transform['cy'] / opt.downscale) if 'cy' in transform else (H / 2)

    intrinsics = np.array([fl_x, fl_y, cx, cy])
    
    for pose in poses:
        pose = torch.squeeze(pose)
        pose = torch.stack([pose])
        result = get_random_rays(opt, pose, intrinsics, H, W)
        results.append(result)
    return results

def load_checkpoint(hyp_model, opt, checkpoint=None, model_only=False):
        ckpt_path = os.path.join(opt.workspace, 'checkpoints')
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{ckpt_path}/ngp_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                
        print(f'Loading checkpoint path from inside utils : {checkpoint}')
        checkpoint_dict = torch.load(checkpoint, map_location=opt.device)
        
        if 'model' not in checkpoint_dict:
            hyp_model.load_state_dict(checkpoint_dict)
            return

        missing_keys, unexpected_keys = hyp_model.load_state_dict(checkpoint_dict['model'], strict=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_index', type=int, default=0, help="render")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--class_choice', type=str, default='chair')
    parser.add_argument('--seed', type=int, default=42)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--load_ckpt', action="store_true", help="if the checkpoint should not be loaded, the checkpoint would be deleted, beware!", required=True)
    parser.add_argument('--eval_interval', type=int, default=5, help="eval once every $ epoch")
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--num_validation_examples', type=int, default=15, help="Number of training samples to take when evaluating compression performance (keep low to speed up training)")
    parser.add_argument('--remove_old', type=bool, default=True, help="Removes checkpoints older than max_keep_ckpt")
    parser.add_argument('--max_keep_ckpt', type=int, default=15, help="Removes checkpoints older than ")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--clip_mapping', action='store_true', help="learn a mapping from clip space to the hypernetwork space")


    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('-b', type=int, default=1, help="batch size")

    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    opt = parser.parse_args()

    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    checkpoints_path = os.path.join(opt.workspace, "checkpoints")

    print(f"Options: {opt}")
    
    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    
    num_examples = 1152 # number of chairs in training set
    
    model = HyPNeRF(opt, num_examples).to(device)
        
    print(model)
        
    model.net.aabb_train = torch.FloatTensor([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]).cuda()
    model.net.aabb_infer = torch.FloatTensor([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]).cuda()
    load_checkpoint(model, opt)
    
    model.eval()
    
    retrieval_index = 0
    
    pred_shape = model.shape_code(torch.LongTensor([retrieval_index]).cuda())
    pred_color = model.color_code(torch.LongTensor([retrieval_index]).cuda())
    pred_params = model.hyper_net(pred_shape, pred_color)
    
    with open(f"{opt.workspace}/poses.pkl", 'rb') as f:
        import pickle
        poses = pickle.load(f)
        results = get_video_rays(poses)
    all_preds = []
    all_preds_depth = []
    
    for result in tqdm.tqdm(results):
        H, W = result['H'], result['W']
        rays_o = result['rays_o'].to(device).squeeze(1)
        rays_d = result['rays_d'].to(device).squeeze(1)
        
        outputs = model.net.render(rays_o, rays_d, staged=True, bg_color=None, perturb=False, force_all_rays=False,params=pred_params,idx=None, **vars(opt))
        
        preds = outputs['image'].reshape(-1, H, W, 3)
        preds_depth = outputs['depth'].reshape(-1, H, W)

        pred = preds[0].detach().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        
        pred_depth = preds_depth[0].detach().cpu().numpy()
        pred_depth = (pred_depth * 255).astype(np.uint8)
        
        all_preds.append(pred)
        all_preds_depth.append(pred_depth)
        
    all_preds = np.stack(all_preds, axis=0)
    all_preds_depth = np.stack(all_preds_depth, axis=0)
    imageio.mimwrite(os.path.join(opt.workspace, f'ngp_{int(time.time())}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
    imageio.mimwrite(os.path.join(opt.workspace, f'ngp_{int(time.time())}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
        