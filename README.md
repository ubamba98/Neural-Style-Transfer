# Neural-Style-Transfer


## Installation(Ubuntu):
 Requirement:
 1. TensorFlow
 2. Keras
 3. Numpy
 4. Scipy
 5. Pretrained-VGG19 Weigths
 
 * Clone this Repo
 * Download [Pretrained-VGG19 Weigths](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) in pretrained-model
 * Run
      ```bash
         pip install --upgrade tensorflow-gpu \
                               keras \
                               numpy \
                               scipy 
      ```
 * Change in Variables.py 'STYLE_IMAGE' and 'CONTENT_IMAGE' for their paths 
 * Open Teminanal and Run
       ```bash
          python3 Training.py
       ```
## Example
<img align="left" width="400" height="300" src='./output/c1.jpg'>

![alt text](./output/c1.jpg "Content Image")
![alt text](./output/S.jpg "Style Image")
![alt text](./output/generated_image.jpg "Generated Image")

## Note:
   IMAGE_WIDTH,IMAGE_HEIGHT,NOISE_RATIO,LEARNINF_RATE,NUM_ITERATIONS can pe changed at Variables.py
   Part of this code is taken from Deeplearning.ai
