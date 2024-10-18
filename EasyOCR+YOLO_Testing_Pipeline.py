#PACKAGES
#Install and import necessary packages
!pip install easyocr
!pip install ultralytics

import cv2
import easyocr
import matplotlib.pyplot as plt
from PIL import Image

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt

# This needs to run only once to load the model into memory
reader = easyocr.Reader(['en'])

#LOAD IMAGES

'''
#Image Location on Google Colab
image_location = '/content/M214_1015935974_0005_0008_B.jpg' #Archive_image_test.jpg
image = Image.open(image_location)
'''

#If not using Colab, get list of all available files in input folder

# List to store image file names
image_files = []

image_files = ['https://repository.library.northeastern.edu/downloads/neu:m04490268?datastream_id=thumbnail_4',
               'https://repository.library.northeastern.edu/downloads/neu:m0446655f?datastream_id=thumbnail_4',
               'https://repository.library.northeastern.edu/downloads/neu:m0449297n?datastream_id=thumbnail_4',
               'https://repository.library.northeastern.edu/downloads/neu:m04493674?datastream_id=thumbnail_4',

]

# Loop through all files in the folder
if image_files == []:
  for file_name in os.listdir(input_folder_location):
      # Check if the file ends with .png or .jpeg
      if file_name.endswith('.png') or file_name.endswith('.jpeg'):
          image_files.append(file_name)
else:
    !pip install requests # install requests if you don't have it
    import requests

#OCR

for image_name in image_files:
  if "http" in image_name:
    image_location = image_name
    response = requests.get(image_location, stream=True)
    response.raw.decode_content = True
    image = Image.open(response.raw)
    file_type = '.png'
    file_name = image_name.replace('https://repository.library.northeastern.edu/downloads/',"")
  else:
    image_location = os.path.join(input_folder_location,image_name) #full file location
    image = Image.open(image_location)
    file_name = os.path.basename(image_location)[:-4]
    file_type = os.path.basename(image_location)[-4:]
    image = cv2.imread(image)

  #get file and generate folder for it
  print(file_name)
  print(file_type)

  #later you will want to have this changed to represent the folder you want it to be
  image_folder = os.path.join(output_folder_location,file_name)
  print("SAVING COPY TO: " + image_folder)
  # Create the folder if it doesn't exist
  if not os.path.exists(image_folder):
      os.makedirs(image_folder)

  # run OCR
  results = reader.readtext(image)

  # show the image and plot the results

  #RUN THROUGH OCR
  plt.figure(figsize=(15,10))
  plt.imshow(image)
  plt.show()
  detected_words = []
  confidence_scores = []

  plt.imshow(image)
  for res in results:

      # bbox coordinates of the detected text
      xy = res[0]
      xy1, xy2, xy3, xy4 = xy[0], xy[1], xy[2], xy[3]

      # text results and confidence of detection
      det, conf = res[1], res[2]

      # show outputs
      plt.plot([xy1[0], xy2[0], xy3[0], xy4[0], xy1[0]], [xy1[1], xy2[1], xy3[1], xy4[1], xy1[1]], 'r-')
      plt.text(xy1[0], xy1[1], f'{det} [{round(conf, 2)}]')

      # Append to lists
      detected_words.append(det)
      confidence_scores.append(conf)

  #create dataframe of results
  df = pd.DataFrame({
      'WORD DETECTED': detected_words,
      'CONFIDENCE SCORE': confidence_scores})

  # Save the image with original name
  image_path = os.path.join(image_folder, file_name + 'marked' + '.png') #could be dynamic and be file_name instead of .png
  plt.savefig(image_path)
  plt.show()

  # Save the pandas DataFrame as 'markup.csv' with index
  csv_path = os.path.join(image_folder, file_name + 'marked' + '.csv')
  df.to_csv(csv_path, index=True)
  print(df)

  print(f"Image saved to {image_path}")
  print(f"DataFrame saved to {csv_path}")

#YOLO

# Load the weights from our repository
model_path = hf_hub_download(local_dir=".",
                             repo_id="armvectores/yolov8n_handwritten_text_detection",
                             filename="best.pt")
model = YOLO(model_path)

# Do the predictions
for image_name in image_files:
  image_location = os.path.join(input_folder_location,image_name) #full file location
  image = cv2.imread(image_location)
  #res = model.predict(source=image, project='.',name='detected', exist_ok=True, save=True, show=True, show_labels=True, show_conf=True, conf=0.05, )
  #res = model.predict(source=image)
  res = model(image)
  res_plotted = res[0].plot()
  #image_location = os.path.join(input_folder_location,image_name) #full file location
  #plt.figure(figsize=(15,10))
  #plt.imshow(plt.imread(image_location))
  #plt.show()

for result in res:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
    print(boxes)



