# Tactile-Texture-Graphic-Generator
a python file for you to generate a 2.5D model base on your pictures for 3D printing

# Step 1
run:
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# Step 2
run: 
python3 tactile_routeB_ams_1.py [your_file_name] \
  --debug_dir debug \
  --k 16 --tol 12 --n_colors 4 \
  --as_parts --fuse_pitch 0.25 --part_gap_mm 0.15 \
  --parts_root parts_out --run_prefix cat \
  --out [filename].stl
