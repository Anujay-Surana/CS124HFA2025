import cv2
print("Python OpenCV版本:", cv2.__version__)
print("是否支持YuNet接口 FaceDetectorYN_create：", hasattr(cv2, "FaceDetectorYN_create"))
assert hasattr(cv2, "FaceDetectorYN_create"), "当前 OpenCV 不支持 YuNet 接口"
print("✅ 检查通过：YuNet 接口可用！")