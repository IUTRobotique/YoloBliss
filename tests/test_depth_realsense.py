import pyrealsense2 as rs
import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt  

pipeline = rs.pipeline() # Create a pipeline
config = rs.config()

# Configurer la résolution (640x480 est la résolution native supportée)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config) # Start streaming

# Attendre plus longtemps pour l'auto-exposition (30 frames ≈ 1 seconde)
print("⏳ Initialisation de la caméra (auto-exposition)...")
for x in range(30):
  pipeline.wait_for_frames()

print("✅ Capture de l'image...")
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

width, height = depth_frame.get_width(), depth_frame.get_height()

color = np.asanyarray(color_frame.get_data())

colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

# Afficher les deux images côte à côte
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Image RGB
axes[0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
axes[0].set_title('Image RGB')
axes[0].axis('off')

# Heatmap de profondeur
axes[1].imshow(colorized_depth)
axes[1].set_title('Heatmap de profondeur')
axes[1].axis('off')

plt.tight_layout()
plt.show()
