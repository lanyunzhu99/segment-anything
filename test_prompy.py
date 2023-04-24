import cv2
import numpy as np
from segment_anything_1 import build_sam, SamPredictor 
predictor = SamPredictor(build_sam(checkpoint="/public2/home/lanyun/pretrain/sam_vit_h_4b8939.pth"))
image = cv2.imread("./test_image/5.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
masks, _, _ = predictor.predict()
print(masks.shape)
print(masks[0, 100, 100])
out = np.ones(masks.shape)
out = out * masks
out = out[0, :, :] * 255
cv2.imwrite('/public2/home/lanyun/sam/segment-anything/1.jpg', np.uint8(out))

