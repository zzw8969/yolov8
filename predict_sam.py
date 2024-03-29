from ultralytics import SAM

# Load a model
model = SAM('sam_b.pt')

# Display model information (optional)
model.info()

# Run inference with bboxes prompt
# model('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709],save=True)
model.predict('ultralytics/assets/zidane.jpg', labels=[0],save=True)

# Run inference with points prompt
# res = model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1],save=True)
# print(res)