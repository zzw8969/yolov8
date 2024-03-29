from ultralytics import SAM

# Load the model
model = SAM('mobile_sam.pt')

# Predict a segment based on a point prompt
# model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1],save=True)
model.predict('ultralytics/assets/zidane.jpg',points=[900, 370] ,save=True)