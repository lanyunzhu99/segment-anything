import cv2
from segment_anything import build_sam, SamPredictor 
predictor = SamPredictor(build_sam(checkpoint="/home/lanyunz/sam/sam_vit_h_4b8939.pth"))
image = cv2.imread("/home/lanyunz/sam_2/segment-anything/test_image/2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
masks, _, _ = predictor.predict()