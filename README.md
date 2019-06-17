# Vegfru Recognition

The goal of the project is to create an energy-efficient inference model on Android to classify fruits and vegetables.

This project takes the [VegFru Dataset](https://pan.baidu.com/s/1boSNcV9) (the download link is hosted in China so it will be quite slow) and retrains MobileNet V2 on it for image classification.

## Getting Started

1. Download the [VegFru Dataset](https://pan.baidu.com/s/1boSNcV9) and extract it under /data/raw
2. Create a virtual environment with `virtualenv env` and activate it
3. Get the dependencies `pip install -r requirements.txt`
4. Run the preprocess shell script with `./preprocess.sh.` Feel free to delete the files in `data/raw` after.
5. Start the transfer learning by running the retrain shell script with `./retrain.sh.` This will take a long time
6. Check to see if the retrain is successful by running `./test_retrained.sh`
7. Convert the model to tflite format by running `./convert.sh`
8. Check to make sure the tflite model is sane by running `./test_tflite.sh`
