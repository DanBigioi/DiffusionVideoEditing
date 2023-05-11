import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import cv2
import os
from tqdm import tqdm
import skimage.io
from PIL import Image
import PIL
import skvideo.io
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def cv2_loader(path):
    return cv2.imread(path)

def pil_loader(path):
    return Image.open(path).convert('RGB')

def pil_loader_audio(path):
    return Image.open(path)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def make_videos(in_folder = 'C:/Users/dabigioi/Documents/stable-diffusion/data/Generated Frames', out_folder = 'C:/Users/dabigioi/Documents/Diffusion_Dataset/CREMA-D/Output_Videos/'):
    images = make_dataset(in_folder)

    # group the images by their video name
    video_groups = {}
    for image in images:
        image_names = os.path.basename(os.path.normpath(image)).split("_")
        video_name = image_names[1]
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(image)

    for video_name, image_group in video_groups.items():
        video_path = os.path.join(out_folder, f"{video_name}.avi")
        print(video_path)

        frame = cv2.imread(os.path.join(in_folder, image_group[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_path, 0, 25, (width, height))

        for image in image_group:
            video.write(cv2.imread(os.path.join(out_folder, image)))

        cv2.destroyAllWindows()
        video.release()


#make_videos()

def combine_audio_video(video_folder =  'C:/Users/dabigioi/Documents/Diffusion_Dataset/CREMA-D/Output_Videos/', audio_folder =  'C:/Users/dabigioi/Documents/Diffusion_Dataset/CREMA-D/Original_Audio/'):

    for video_name in os.listdir(video_folder):
        video = VideoFileClip(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Output_Videos/"+video_name)
        audio = AudioFileClip(r"C:\Users\dabigioi\Documents\stable-diffusion\data\GRID\S1_Wavs/"+video_name[:-4]+".wav")

        # Combine the audio with the video
        new_audio = CompositeAudioClip([audio])
        video = video.set_audio(new_audio)

        video.write_videofile(
            r"C:/Users/dabigioi/Documents/Diffusion_Dataset/CREMA-D/Output_Videos_With_Audio/"+video_name[:-4]+".mp4", codec="libx264",
            audio_codec="aac", fps=25)

#combine_audio_video()
