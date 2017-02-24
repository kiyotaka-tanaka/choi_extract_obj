local image_dir = './bev_pred'
local out_dir = './features_predict_bev'
local prototxt = 'VGG_ILSVRC_16_layers_deploy.prototxt'
local caffemodel = 'VGG_ILSVRC_16_layers.caffemodel'
local layer_to_extract = 37 -- 39=fc8, 37=fc7, 31=pool5
local batch_size = 2
local image_size = 224
local gpu_device = 1 
local mean_image = {103.939, 116.779, 123.68}
local ext = 'h5' -- 'h5' or 't7'
local force = true

assert(image_dir:sub(image_dir:len()) ~= '/', 'image_dir should not end with /')

-- load dependencies
require 'cutorch'     -- CUDA tensors
require 'nn'          -- neural network package
require 'cudnn'       -- fast CUDA routines for neural networks
require 'loadcaffe'   -- loads models from Caffe
require 'paths'       -- utilities for reading directories 
require 'image'       -- reading/processing images
require 'hdf5'        -- writing hdf5 files
require 'xlua'        -- for progress bar

-- set GPU device
-- check which GPUs are free with 'nvidia-smi'
-- first GPU is #1, second is #2, ...
cutorch.setDevice(gpu_device)

-- loads model from caffe
local model = loadcaffe.load(prototxt, caffemodel, 'cudnn');

model:evaluate() -- turn on evaluation model (e.g., disable dropout)
model:cuda() -- ship model to GPU

--print(model) -- visualizes the model
--print('extracting layer ' .. layer_to_extract)

-- tensor to store RGB images on GPU 
local input_images = torch.CudaTensor(batch_size, 3, image_size, image_size)

-- utility function to check if file exists
function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end

-- read all *.jpg files in the 'image_dir', and store in the array 'filepaths'
-- we recursively search the image dir
local filepaths = {};
local feat_paths = {}
function find_images(searchdir)
  print('searching ' .. searchdir)
  -- search sub directories
  for f in paths.iterdirs(searchdir) do
    find_images(searchdir .. '/' .. f)
  end
  -- add files in this directory
  for f in paths.iterfiles(searchdir) do
    local f_full = searchdir .. '/' .. f
    local feat_full = out_dir .. '/' .. f_full:sub(image_dir:len()+2) .. '.' .. ext

    -- add to work queue if not done, or force is set
    if force or not file_exists(feat_full) then
      table.insert(filepaths, f_full)
      table.insert(feat_paths, feat_full)
    end
  end
end
find_images(image_dir) -- start recursion
print('found ' .. #filepaths .. ' images')

-- function to read image from disk, and do preprocessing
-- necessary for caffe models
function load_caffe_image(impath)
  local im = image.load(impath)                 -- read image
  im = image.scale(im, image_size, image_size)  -- resize image
  im = im * 255                                 -- change range to 0 and 255
  im = im:index(1,torch.LongTensor{3,2,1})      -- change RBB --> BGR

  -- subtract mean
  for i=1,3 do
    im[{ i, {}, {} }]:add(-mean_image[i])
  end

  return im
end

-- function to run feature extraction
function extract_feat(size, last_id)
  -- do forward pass of model on the images
  model:forward(input_images)

  -- read the activations from the requested layer
  local feat = model.modules[layer_to_extract].output

  -- ship activations back to host memory
  --print(feat)
  feat = feat:float()
  --print(feat)
  -- save feature for item in batch
  for i=1,size-1 do
    -- make output directory if needed
    paths.mkdir(paths.dirname(feat_paths[i+last_id-1]))

    if ext == 'h5' then -- save hdf5 file
      local hdf5_file = hdf5.open(feat_paths[i+last_id-1], 'w')
      hdf5_file:write('feat', feat[i])
      hdf5_file:close()

    elseif ext == 't7' then -- save torch7 file
      torch.save(feat_paths[i+last_id-1], feat[i])

    else
      assert(false, 'unknown filetype')
    end
  end
end

-- current index into input_images
local counter = 1

-- last time we modified 
local last_id = 1

-- loop over each image
for image_id, filepath in ipairs(filepaths) do
  xlua.progress(image_id, #filepaths) -- display progress

  -- read image and store on GPU
  input_images[counter] = load_caffe_image(filepath)

  -- once we fill up the batch, extract, and reset counter
  counter = counter + 1
  if counter > batch_size then
    extract_feat(counter, last_id) -- extract
    counter = 1                    -- reset counter
    last_id = image_id+1
    input_images:zero()            -- for sanity, zero images
  end
end

-- one last time for end of batch
if counter > 1 then
  extract_feat(counter, last_id)
end
