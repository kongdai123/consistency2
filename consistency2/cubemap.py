import torch
import nvdiffrast.torch as dr
import numpy as np 


def create_meshgrid(
    height,
    width,
    normalized_coordinates=True,
    device=torch.device("cpu"),
    dtype=torch.float32,
):
    xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    base_grid = torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2



def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack(
        [(cent[0] - i) / focal[0], (cent[1] - j) / focal[1], torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions


class CubeMap():
    def __init__(self, env, device = "cuda", H = 512, W = 512, fov = 40):
        self.device = device
        self.env = env.to(torch.float32) #[1, 6, H, W, C]

        self.get_samples(H, W, fov)


    def get_samples(self, H, W, fov):
        focal_length = [
            1 / np.tan(fov / 2 * np.pi / 180) * H / 2,
            1 / np.tan(fov / 2 * np.pi / 180) * W / 2,
        ]
        ray_directions = get_ray_directions(H, W, focal_length).to(self.device)
        ray_directions /= torch.sum(ray_directions**2, -1, keepdim=True)**0.5 
        ray_directions = ray_directions[None,] #[1, H, W, 3]
        ray_directions[..., 0] *= -1
        self.ray_directions = ray_directions
        # return ray_directions # [1, H, W, 3]

    def texture(self, glctx, r_rot_all, mip_level_bias = None):
        H, W = self.ray_directions.shape[1], self.ray_directions.shape[2]
        ray_out = self.ray_directions.reshape((1, H * W, 3))
        ray_out = torch.matmul(ray_out, r_rot_all.transpose(1, 2))
        ray_out /= torch.sum(ray_out**2, -1, keepdim=True)**0.5 
        ray_out = ray_out.reshape((ray_out.shape[0], H, W, 3))
        ray_out = ray_out.to(torch.float32)
        #only works with float 32
        if mip_level_bias is None:
            mip_level_bias = 0 * torch.ones_like(ray_out[..., 0])
        
        color = dr.texture(self.env, ray_out,mip_level_bias =  mip_level_bias, uv_da = None, filter_mode='linear-mipmap-linear', boundary_mode='cube')

        return color #[b, H, W, C]