import argparse

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay


mp_face_mesh = mp.solutions.face_mesh

def main(name):
  # breakpoint()
  # load image
  # image = cv2.imread(name)
  image = np.array(Image.open(name))
  height, width = image.shape[:2]

  # infer face mesh
  with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5
  ) as face_mesh:
    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  mesh = results.multi_face_landmarks[0]

  # flatten and scale mesh
  flat_mesh = np.array([(coords.x, coords.y) for coords in mesh.landmark])
  # breakpoint()
  scaled_mesh = flat_mesh * np.array([width, height])
  
  # compute face mask
  delaunay = Delaunay(scaled_mesh)
  coords = list(np.ndindex((width, height)))
  simplices = delaunay.find_simplex(coords)
  mask = (simplices >= 0).reshape((width, height)).transpose()

  # invert, save, and return mask
  mask = Image.fromarray(np.logical_not(mask))
  mask.save("../mask.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("path")
  args = parser.parse_args()
  main(args.path)
