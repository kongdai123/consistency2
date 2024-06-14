import torch
import numpy as np 

from pytorch3d.transforms import euler_angles_to_matrix


def projection(x=0.1, n=1.0, f=50.0):
    #https://gamedev.stackexchange.com/questions/120338/what-does-a-perspective-projection-matrix-look-like-in-opengl
    #x = n * tan(fov/2)
    #n near plane
    #f far plane
    return np.array([[n/x,    0,            0,              0],
                     [  0,  -n/x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)
def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m


def process_multi_items(arg, l):
    if isinstance(arg, list):
        assert (len(arg) == l), "arg length mismatch"
        return arg
    
    # Check if a variable is a string
    if isinstance(arg, str) or isinstance(arg, int) or isinstance(arg, float):
        return [arg] * l
    
    raise Exception("input type mismatch")


def positive_modulo(dividend, divisor):
    return (dividend % divisor + divisor) % divisor

class ViewDataset:
    def __init__(self,
                 elev,
                 azim,
                 dist,
                 prompts = "",
                 negative_prompts = "",
                 view_dep_prompts = True, 
                 device = "cuda",
                 Z_near = 0.1,
                 Z_far = 10,
                 fov = 40):
        self.azim = [positive_modulo(a, 360) for a in azim]
        self.elev = [positive_modulo(e + 90, 181) - 90 for e in elev]
        self.len = len(azim)
        self.dist = process_multi_items(dist, self.len)
        self.prompts = process_multi_items(prompts, self.len)
        self.prompts_ori = process_multi_items(prompts, self.len)
        self.negative_prompts = process_multi_items(negative_prompts, self.len)
        self.view_dep_prompts = view_dep_prompts
        if view_dep_prompts:
            for i in range(self.len):
                self.prompts[i] = self.prompts_ori[i] + ", " + self.get_view_dep_prompt(self.elev[i], self.azim[i])

        self.device = device
        
        dist = 3.5
        self.Z_near = Z_near
        self.Z_far = Z_far
        self.fov = process_multi_items(fov, self.len)
        self.process_intrinsics()


        self.rot_all = self.get_rot_all()
        self.r_mvp_all = self.get_mvp_all()
        self.rot_env_all = self.get_rot_env_all()
    
    def process_intrinsics(self):
        # Modelview and modelview + projection matrices.
        self.proj = []
        
        for i in range(self.len):
            x = self.Z_near * np.tan(self.fov[i]/2 * np.pi/180)
            proj  = projection(x=x , n=self.Z_near, f=self.Z_far)
            self.proj.append(torch.tensor(proj).to(self.device))
        
        self.proj = torch.stack(self.proj, dim = 0)
    

    def update_views(self, elev = None, azim = None, dist = None, fov = None):
        
        if elev is not None:
            assert(len(elev) == self.len)
            self.elev = [positive_modulo(e + 90, 181) - 90 for e in elev]
        if azim is not None:
            assert(len(azim) == self.len)
            self.azim = [positive_modulo(a, 360) for a in azim]
        
        if (elev is not None) or (azim is not None):
            if self.view_dep_prompts:
                for i in range(self.len):
                    self.prompts[i] = self.prompts_ori[i] + ", " + self.get_view_dep_prompt(self.elev[i], self.azim[i])

        if dist is not None:
            self.dist = process_multi_items(dist, self.len)
        
        if fov is not None:
            self.fov = process_multi_items(fov, self.len)
            self.process_intrinsics()
        
        self.rot_all = self.get_rot_all()
        self.r_mvp_all = self.get_mvp_all()
        self.rot_env_all = self.get_rot_env_all()

    @torch.no_grad()
    def get_rot_matrix(self, idx):

        if idx >= self.len:
            raise Exception("View index out of range")
        
        euler_angles = torch.tensor([np.pi * self.azim[idx]/ 180, np.pi * self.elev[idx]/ 180, 0]).to(self.device)
        R = euler_angles_to_matrix(euler_angles, convention= "YXZ").to(torch.float32)
        return R.inverse() #[3, 3]
    
    
    @torch.no_grad()
    def get_rot_env_matrix(self, idx):

        if idx >= self.len:
            raise Exception("View index out of range")
        
        euler_angles = torch.tensor([np.pi * self.elev[idx]/ 180, np.pi * self.azim[idx]/ 180, 0]).to(self.device)
        R = euler_angles_to_matrix(euler_angles, convention= "XYZ").to(torch.float32)
        return R.inverse() #[3, 3]
    
    def get_rot_all(self):
        return torch.stack([self.get_rot_matrix(i) for i in range(self.len)], dim=0) #[l, 3, 3]
    
    def get_rot_env_all(self):
        return torch.stack([self.get_rot_env_matrix(i) for i in range(self.len)], dim=0) #[l, 3, 3]

    @torch.no_grad()
    def get_mvp_matrix(self, idx):
        if idx >= self.len:
            raise Exception("View index out of range")
        
        r_mv  = torch.tensor(translate(0, 0, -self.dist[idx])).to(self.device)
        r_mv[:3,:3] = self.get_rot_matrix(idx)
        
        r_mvp = torch.matmul(self.proj[idx], r_mv).to(torch.float32)
        
        return r_mvp #[4, 4]
    
    @torch.no_grad()
    def get_mvp_all(self):

        return torch.stack([self.get_mvp_matrix(i) for i in range(self.len)], dim=0) #[l, 4, 4]

    def get_view_dep_prompt(self, elev, azim):

        # elev [-90, 90]
        # azim [0, 360]
        res = ""

        side_thresh = 30
        front_thresh = 60
        elev_thresh_extreme = 60
        elev_thresh = 30

        if elev <= - elev_thresh:
            res = "overhead "
        elif elev >= elev_thresh:
            res = "botttom "
        else:
            pass
        
        if elev <= elev_thresh_extreme and elev >= -elev_thresh_extreme:
            if azim >= side_thresh and azim <= 180 - side_thresh:
                res = res + "left "
            elif azim >= side_thresh + 180 and azim <= 360 - side_thresh:
                res = res + "right "
            else: 
                pass
            
            if azim <= front_thresh or azim >= 360 - front_thresh:
                res = res + "front "
            elif azim >= 180 - front_thresh and azim <= 180 + front_thresh:
                res = res + "back "
            else:
                pass  

        res = res + "view"
        return res


def round_view_dataset(
    start_angle = -30,
    dist_vis = 3.5,
    fov = 40,                 
    prompts = "",
    negative_prompts = ""
    ):
    
        views_vis = 360
        elev_angs_vis = [0] * views_vis
        azim_angs_vis = [i * (360/views_vis)  for i in range(views_vis)]
        azim_angs_vis = [a + start_angle for a in azim_angs_vis]

        dist_vis = [3.5] * views_vis
        vd_vis = ViewDataset(elev_angs_vis, azim_angs_vis, dist_vis, prompts,negative_prompts, fov = fov)
        return vd_vis
