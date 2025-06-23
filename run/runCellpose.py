# Set the path to store cellpose models.
import numpy as np
from cellpose import models, core, io, plot, utils
from pathlib import Path
from tqdm import trange
from natsort import natsorted
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

# =  = = = = = = = = = = = = = = = = = #
#   SETTINGS AND HYPERPARAMETERS
# =  = = = = = = = = = = = = = = = = = #
flow_threshold      = 0.9
cellprob_threshold  = 0.0
tile_norm_blocksize = 0.0
compact_threshold   = 0.125
niter               = 12000
min_size            = 20

# =  = = = = = = = = = = = = = = = = = #
#   MODEL SETUP
# =  = = = = = = = = = = = = = = = = = #
io.logger_setup(cp_path='../'*(len(Path.home().parts)-1)+str(Path.cwd()), logfile_name="cellpose.log", ) # run this to get printing of progress

# Record the GPU model
if core.torch.cuda.is_available():
    device = core.torch.cuda.current_device()
    gpu_name = core.torch.cuda.get_device_name(device)
    print(f"Current GPU: {gpu_name}")

# Load the neural network model
model = models.CellposeModel(gpu=core.use_gpu())

dir = Path("celldetection/")
if not dir.exists():
  raise FileNotFoundError("directory does not exist")


# =  = = = = = = = = = = = = = = = = = #
#   FILE READING
# =  = = = = = = = = = = = = = = = = = #

# list all files
image_ext = ".png"
files = natsorted([f for f in dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name and "_overlay" not in f.name])

if(len(files)==0):
  raise FileNotFoundError("no image files found, did you specify the correct folder and extension?")
else:
  print(f"{len(files)} images in folder:")

for f in files: print(f.name)

# Create output directory
os.makedirs(os.path.join(f.parent, 'output'), exist_ok=True)

# =  = = = = = = = = = = = = = = = = = #
#   NEURAL NETWORK
# =  = = = = = = = = = = = = = = = = = #

np.random.seed(4096) # just used for coloring
for i in trange(len(files)):
    # Select the file and output path
    file = files[i]
    out_path = file.parent / 'output' / file.stem
    # Read the image and convert to grascale
    img = io.imread(file).mean(axis=2).astype(np.uint8)

    # Run the neural network
    masks, flows, styles = model.eval(img, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                  normalize={"tile_norm_blocksize": tile_norm_blocksize}, niter = niter) 

    # > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > #
    # Filter out small or ultra-high aspect ratio masks, which aren't reliable
    masks = utils.fill_holes_and_remove_small_masks(masks, min_size=min_size)
    
    # Obtain the list of masks and get areas too. Also get perimeters from cellpose utils
    mask_list, areas = np.unique(masks, return_counts = True)
    perimeters = utils.get_mask_perimeters(masks)

    # Filter out masks that are too skinny
    compactness = np.append([1.0], 4*np.pi*areas[1:]/perimeters**2)
    masks[np.vectorize(dict(zip(mask_list, compactness)).get)(masks) < compact_threshold] = 0
    mask_list = mask_list[compactness >= compact_threshold]
    areas     = areas[compactness >= compact_threshold]

    # Extract outlines and edges
    outlines  = utils.masks_to_outlines(masks)
    edges     = utils.masks_to_edges(masks)
    # Flow vector properties
    flow_mag  = np.sqrt(flows[1][0]**2 +flows[1][1]**2)
    flow_ang  = np.arctan2(flows[1][0], flows[1][1]) + np.pi
    # shading from upper left
    shade     = 0.5*(np.sin(flow_ang - np.pi*3/6) + 1)

    # Centers of mass
    centers = np.zeros_like(img, dtype=bool)
    center_list = np.zeros([len(mask_list), 2]) # N x 2 list of cell centers
    for i in range(len(mask_list)):
        cy, cx = center_of_mass(masks == mask_list[i])
        centers[int(round(cy)), int(round(cx))] = True
        center_list[i] = [cx, cy]

    # Create the overlay using Hue-Saturation-Value color
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    # Hues determined randomly by the masks
    HSV[:, :, 0] = np.vectorize({val: np.random.random() for val in mask_list}.get)(masks)
    
    # Saturation set to 1 where cells exist, excluding outlines
    HSV[:, :, 1] = (masks!=0) * (1 - outlines)
    # Apply a small amount of randomness to the saturation for each marker
    # HSV[:, :, 1] *= (1 - 0.25*np.vectorize({val: np.random.random() for val in mask_list}.get)(masks))
    # Shading is applied via saturation
    HSV[:, :, 1] *= (0.6 + 0.4*shade)

    # Value (brightness determined by original image)
    HSV[:, :, 2] = (img/img.max())
    # Scale the brightness
    HSV[:, :, 2] = (HSV[:, :, 2]**0.6).astype(np.float32)
    # Apply a small amount of randomess to the darkness of each marker
    HSV[:, :, 2] *= (1 - 0.25*(masks!=0)*(1-outlines)*np.vectorize({val: np.random.random() for val in mask_list}.get)(masks)) 

    # Convert to RGB for the final overlay image
    overlay = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    
    # Create the outlines image
    outlined = np.repeat(img[..., np.newaxis], 3, axis=-1)
    outlined[outlines] = [225, 0, 0]
    outlined[centers]  = [0, 225, 0]

    # Save the processed images
    io.imsave(str(out_path) + "_overlay.png", overlay)
    io.imsave(str(out_path) + "_flows.png", flows[0])
    io.imsave(str(out_path) + "_outlines.png", outlined)

    # Save the data
    header = "Cell, Area (cm^2), X Location (cm), Y Location (cm)"
    CellData = np.column_stack((mask_list, areas, center_list[:,0], center_list[:,1]))[1:,:]
    np.savetxt(str(out_path) + '_cell_data.csv', CellData, delimiter=',', header=header, comments='', fmt='%f')
