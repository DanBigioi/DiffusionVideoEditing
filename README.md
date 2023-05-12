# DiffusionVideoEditing

Official project repo for the paper "Speech Driven Video Editing via an Audio-Conditioned Diffusion Model" 


https://github.com/DanBigioi/DiffusionVideoEditing/assets/44242832/474b1c06-daf7-4d79-ad30-247c0269b68c


![network_diagram](https://github.com/DanBigioi/DiffusionVideoEditing/assets/44242832/f71dbba2-e1da-49ac-a388-8a7499816897)

Shoutout to https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models ! Most of the code in this repo is taken from there. It's a really good implementation of the Palette image2image paper, so go check it out!

Additionally, make sure to check out the repo of our co-authors https://github.com/MStypulkowski/diffused-heads ! They utilise diffusion models for the related task of talking head generation.

You can check out some of our results on the project page found here: https://danbigioi.github.io/DiffusionVideoEditing/

To read our paper, click on this link: https://arxiv.org/abs/2301.04474 . Note that our paper is currently under review. 

# Set Up

This code was tested on using Python version 3.8.12 , and CUDA 11.0 . The listed requirements in the requirements.txt file are what we used, however one could use a more recent version of both CUDA and Pytorch, just note that you may have to deal with some dependency issues that should be easily solveable. If you run into any issues running the code, feel free to post them here. 

To install the repository run the following commands: 

```
  git clone https://github.com/DanBigioi/DiffusionVideoEditing.git

  cd DiffusionVideoEditing 

  pip install -r requirements.txt
```

# Dataset Download

Next, from the root directory of the repo, create the following directories:

```
  mkdir datasets

  cd datasets

  mkdir CREMAD

  mkdir S1
  
```

For training/testing the multi-speaker model on the CREMA-D dataset, download the following zipped files, unzip them, and put them in the CREMAD folder: 

> - Audio Features MultiSpeaker: https://drive.google.com/file/d/1Z-WmCNvxNcfUiv2km641EGqRFNqavIKP/view?usp=sharing

> - Cropped Frames Train: https://drive.google.com/file/d/1Q8ePAabAXu6nMyJLOJncR72fo0vtsQ1p/view?usp=sharing

> - Cropped Frames Test: https://drive.google.com/file/d/10bG5Zm1-cu71CJBfZcT7JRkEf3sxdWCq/view?usp=sharing

For training/testing the single speaker model on identity S1 of the GRID dataset, download the following zipped files, unzip them, and put them in the S1 folder: 

> - Audio Features S1: https://drive.google.com/file/d/1vH8at1mrmOIe-Ljip-tO2BakLTjRRdqt/view?usp=sharing

> - Cropped Frames Train: https://drive.google.com/file/d/1v4tv4eZYAlswKvo4YOZpMNnra3TEj6KG/view?usp=sharing

> - Cropped Frames Test: https://drive.google.com/file/d/1RU20QAa6H9HemSScWRbKfsnpeIqwUm8I/view?usp=sharing

Also download the following folder called "Generated_Frames", unzip it, and place it in the datasets folder: 

> - Generated_Frames: https://drive.google.com/file/d/1Cxw5pUMoIs3rPLEGu_jQDjGO-0PGWFZo/view?usp=sharing

Its purpose is to store the frames generated during inference, and we have configured it to contain frame 0 of each video in the CREMA-D test set. 

# Model Weights Download

To download the pretrained model weights for our multi-speaker model, download the following zipped folder, unzip it, and place it in the weights folder:

> - MultiSpeaker_Cremad_Epoch_735: https://drive.google.com/file/d/1klsQXUKQtq-pPPqPUIktcbXK0FE-31jP/view?usp=sharing

To download the pretrained model weights for our single-speaker model, download the following .ckpt file and place it in the weights folder:

> - SingleSpeaker_S1Grid_Epoch_2050: TO BE DONE

# Training

In order to train your model from scratch on the preprocessed files we provide, simply run the following command: 

```
python run.py -c config/audio_talking_heads.json -p train 
```

Just make sure you modify the config file to specify how many GPUs you want to use, and to point towards where you stored the train set files that you downloaded above. Additionally, modify the config file to say "resume_state": null . If you wish to resume training from the pretrained checkpoint we provide, simply modify the config to say "resume_state": "weights/multi-speaker_checkpoint/735". 


# Data Processing For Custom Training

To train a model from scratch on your own custom dataset, prepare the data in the following way: 

#### Video Processing

> - Each video within your dataset should have a unique name. For example, in the crema-D dataset, we follow the convention "ID_Caption_Emotion_Volume.mp4" 
> - Crop your videos around the facial region, using a 1:1 aspect ratio. We recommend this facial alignment tool: https://github.com/DinoMan/face-processor (Note that this step is optional, but we recommend it for better results)
> - Once you have cropped your videos, resize them to 128x128 (again optional, but to save on computational power, we recommend going with a smaller video size). This can be done by using the "rescale_videos()" method we provide in utils/data_preprocessing.py. 
> - From each video you intend to use to train your model, extract the individual frames, saving them as png files. For videos within the Crema-D dataset, we use the following naming convention "ID_Caption_Emotion_Volume_000.png", "ID_Caption_Emotion_Volume_001.png", etc. This can be done using the "extract_video_frames()" method we provide in utils/data_preprocessing.py. 
> - Take a look at how we process our own data to gain a better understanding, this can be found in DiffusionVideoEditing/datasets/CREMAD/Cropped_Frames_Train/. You should end up with a single folder containing all the frames from your video dataset. 

#### Audio Processing

> - You should have a separate folder containing the audio files associated with each video. These audio files should be named the same as the videos, eg.  "ID_Caption_Emotion_Volume.wav" instead of  "ID_Caption_Emotion_Volume.mp4". 
> - Use the "audio_data_extraction()" method we provide in utils/data_preprocessing.py in order to compute the mel spectral features that will be used to train the model. This method ensures audio length matches video length, and creates the mel spectrogram feature, saving it as a numpy file. Make sure to modify all paths to point to your own data. You should end up with a folder containing numpy files of the spectrgrams. 

Once your data is correctly processed, you can now start training the model from scratch, just don't forget to modify the config/audio_talking_heads.json file to point towards your processed video frames and audio files. 

# Inference

Before running inference, make sure you download and unzip the Generated_Frames folder and placed it within the datasets folder. This folder will store the frames generated by the model. 

Then, for the video you wish to modify, process it in the same way you would process the training set videos. That is, crop it, resize it, and extract the individual frames as described above. One extra step is needed though, move frame 0 into the Generated_Frames_Folder, and rename it from "ID_Caption_Emotion_Volume_000.png" to "Out_ID_Caption_Emotion_Volume_000.png". The audio is processed the same, just ensure that the name of the audio file is the same as the name of the original video ie. "ID_Caption_Emotion_Volume_000.wav". 

This is necessary because our model is autoregressive and requires a "seed" frame to start the generation process. This is provided by frame 0. The "Generated_Frames" folder you unzipped should contain 820 such seed frames, each one corresponding to a video within the CREMA-D test set. Additionally, we already preprocess the videos within the CREMA-D test set for you, they can be found in the folder datasets/CREMAD/Cropped_Frames_Test. 

To run inference, simply run 

```
python run.py -c config/audio_talking_heads_inference.json -p infer 
```

Note that inference can only be run on a single GPU with a batch size of 1, and 0 num_workers. This is because the method is entirely autoregressive and relies on the previously generated frame to generate the following one. Make sure you modify the inference section of the json to point towards the frames you want to modify, and audio. 

**We are currently working on preparing an easier to use inference script, so stay tuned for updates there!**

Once inference finishes running, the folder datasets/Generated_Frames will contain all the newly generated frames. To turn them into a video, use the two methods we provide in utils/make_video.py , namely the make_videos() and combine_audio_video() methods. 













