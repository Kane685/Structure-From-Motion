# How to Run Code from Week2 to Week4


## Set Environment

```bash
conda create -n gf4 python=3.10 -y  
conda activate gf4 
pip install numpy
pip install opencv-python
pip install plotly
pip install open3d
pip install scipy
```


## Week2

### Run Pair Mode

```bash
python week2/week2_pipeline.py \
  --image1 path/to/image_01.jpg \
  --image2 path/to/image_08.jpg \
  --output-dir week2/output/pair_debug \
  --max-features 4000 \
  --ratio 0.75
```

### Run Dataset Mode

```bash
python week2/week2_pipeline.py \
  --image-dir week2/data/good_subset/images \
  --output-dir week2/output/good_subset \
  --max-images 20 \
  --max-features 4000 \
  --ratio 0.75
```


## Week3

### For Pair Only Debugging

```bash
python week3/week3_pipeline.py \
  --image1 path/to/good_01.jpg \
  --image2 path/to/good_08.jpg \
  --output-dir week3/output/pair_only \
  --week2-dir week2 \
  --max-features 4000 \
  --ratio 0.75
```

### For a Good Image Pair plus a Third Image

```bash
python week3/week3_pipeline.py \
  --image1 path/to/good_01.jpg \
  --image2 path/to/good_08.jpg \
  --image3 path/to/good_12.jpg \
  --output-dir week3/output/three_view \
  --week2-dir week2 \
  --max-features 4000 \
  --ratio 0.75
```


## Week4

### Set Proper Environment for Open3d Visualization only for WSL

```bash
sudo apt install x11-apps
unset WAYLAND_DISPLAY
export DISPLAY=:0
export XDG_SESSION_TYPE=x11
export LIBGL_ALWAYS_SOFTWARE=1
```

### Reconstruction without Final Filtering

```bash
python week4/week4_pipeline.py \
  --image-dir path/to/image_dataset \
  --output-dir path/to/output \
  --final-filter 0
```

### Reconstruction with Final Filtering

```bash
python week4/week4_pipeline.py \
  --image-dir path/to/image_dataset \
  --output-dir path/to/output \
  --final-filter 1
```

### Visualization from Exported Point Cloud

```bash
python week4/visualization.py \
  --result-path path/to/point_cloud_file
```