import imageio
from consistency2.utils import *
import matplotlib.pyplot as plt
from PIL import Image


def output_video(path,
                 mesh, 
                 view_dataset,
                 glctx,
                 rendersettings,
                 tex= None,
                 cm = None,
                 diff = None,
                 fps = 25,
                 quality = 8,
                 save_dir = None):
    r_mvp_all = view_dataset.get_mvp_all()
    r_rot_all = view_dataset.get_rot_all()
    r_rot_env_all = view_dataset.get_rot_env_all()
    
    all_preds = []
    if rendersettings.latent:
        assert(diff is not None)

    if save_dir is not None:
        os.makedirs(f"{save_dir}/360", exist_ok=True)
    
    for i in range (view_dataset.len):
        r_mvp = r_mvp_all[i: i + 1]
        r_rot = r_rot_all[i: i + 1]
        r_rot_env = r_rot_env_all[i: i + 1]
        color, rast_out = mesh.render(glctx, r_mvp, rendersettings, tex = tex)
        
        
        if cm is not None:
            object_mask = (rast_out[..., -2]  != 0)
            color_bg = cm.texture(glctx, r_rot_env)
            
            if rendersettings.latent:
                color_bg = diff.encode_imgs(color_bg, permute = True)
                color = torch.where(object_mask[:,None, :,:], color, color_bg)
            else:
                color = torch.where(object_mask[:,:,:, None], color, color_bg)

        if rendersettings.latent:
            color = diff.decode_latents(color, permute = True)
        
        all_preds.append((color.detach().clamp(0,1).cpu().numpy()[0] * 255).astype("uint8"))

    all_preds = np.stack(all_preds, axis=0)

    dump_vid = lambda video, name: imageio.mimwrite( f"{name}.mp4", video, fps=fps, quality=quality, macro_block_size=1)

    dump_vid(all_preds, path)

    if save_dir is not None:
        for i in range(view_dataset.len):
            image = Image.fromarray(all_preds[i])
            image.save(f"{save_dir}/360/{i}.png")



def plt_vis(path, tensor, colorbar = True):
    
    plt.imshow(tensor.detach().cpu().numpy())
    if colorbar:
        plt.colorbar()
    plt.savefig(path,  bbox_inches='tight')
    plt.show()
    plt.clf()


def save_combined(save_dir, latents, vd, pipe):
    views = vd.len

    fig, axes = plt.subplots(2, views//2, figsize=(24,12))

    # Flatten the 2D axes array to simplify indexing
    axes = axes.flatten()

    # Loop through each subplot, plot amipn image, and set the header
    img_all = []
    for i, ax in enumerate(axes):
        # Plot the image
        img = pipe.decode_latents(latents[i: i + 1], permute = True)[0].to(torch.float32)
        img_all.append(img)

        elev = vd.elev[i] 
        
        azim = vd.azim[i]
        
        save_torch_img(img, f"{save_dir}/{elev}_{azim}.png")

        ax.imshow(img.cpu().numpy()) 
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig(f"{save_dir}/combined.png",bbox_inches = 0.0)

    # Show the plot
    plt.show()
    
    img_all = torch.stack(img_all, axis = 0)

    return img_all