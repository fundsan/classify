Sushi Sandwich Classifier
====================
##Instructions

My model uses a pretrained convolution network, the VGG-16 graph through the Keras wrapper library. This choice was made due to the lack of data that I was able to collect. First download the model file at this website: https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

put the file in the repository directory

I collected 2800 images all together (including using some of the files given to me). you can download the file and put it in the repository directory here: https://drive.google.com/open?id=0B5ex_C5_-lppX1hZS3hFWTQ0Y1E

check the requirements.txt file or run it in your environment through using pip 
> pip install -r /path/to/requirements.txt


To train the model type in a terminal while in the correct directory: 
 > python train.py

On the first run, the model will gather the pretrained graph and run the data through the model and save it to a file. The script will take some time to save the bottleneck files (about 20-30 minutes), but this will greatly speed up the training. You should see graphs produced similar to the ones below once the regular training has finished.

To test on the model, type: 
> python test.py <directory or file name>
