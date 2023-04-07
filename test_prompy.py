
from segment_anything import build_sam, SamPredictor 
predictor = SamPredictor(build_sam(checkpoint="/home/lanyunz/sam/sam_vit_h_4b8939.pth"))
predictor.set_image("/home/lanyunz/sam_2/segment-anything/test_image/1.jpg")
masks, _, _ = predictor.predict("birds in the image")