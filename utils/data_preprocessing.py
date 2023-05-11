from PIL import Image
import os
import numpy as np
import ntpath
import librosa
import skimage
import skvideo.io
import librosa.display
import cv2
import skimage.io
import tqdm

def pil_loader(path):
    return Image.open(path)

def rescale_frame(frame_input):
    width = int(128)
    height = int(128)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

def image_resize(image, width = 128, height = 128, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def rescale_videos(video_path):
    cap = cv2.VideoCapture(video_path)

    (h, w) = (128,128)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Resized_64_64/"+ntpath.basename(video_path)[:-4]+".mp4",
                             fourcc, 25.0,
                             (w, h), True)


    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break

        rescaled_frame = rescale_frame(frame)
        # write the output frame to file
        writer.write(rescaled_frame)


    cv2.destroyAllWindows()
    cap.release()
    writer.release()

#for videos in os.listdir(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Original_Video"):
#    rescale_videos(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Original_Video/"+videos)

def audio_data_extraction():
    for video_name in tqdm(os.listdir(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Original_Video")):

        y, sr = librosa.load(
            r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Original_Audio/" + video_name[:-4] + '.wav',
            sr=16000)

        print(len(y))

        cap = cv2.VideoCapture(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Original_Video/" + video_name)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_audio_signal_length = (num_frames * 640) - 1

        if target_audio_signal_length < y.shape[0]:
            y = y[:target_audio_signal_length]

            M = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=2048, win_length=640, hop_length=320, n_mels=256,
                                               fmax=8000)
            M_db = librosa.power_to_db(M, ref=np.max)
            np.save(
                r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Audio_16Khz_Numpy/" + video_name[:-4] + '.npy',
                M_db)


        elif target_audio_signal_length > y.shape[0]:

            difference = target_audio_signal_length - y.shape[0]
            zeros = np.zeros((difference,))
            y = np.concatenate((y, zeros))

            M = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=2048, win_length=640, hop_length=320, n_mels=256,
                                               fmax=8000)
            M_db = librosa.power_to_db(M, ref=np.max)
            np.save(
                r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Audio_16Khz_Numpy/" + video_name[:-4] + '.npy',
                M_db)

        elif target_audio_signal_length == y.shape[0]:

            M = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=2048, win_length=640, hop_length=320, n_mels=256,
                                               fmax=8000)
            M_db = librosa.power_to_db(M, ref=np.max)
            np.save(
                r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Audio_16Khz_Numpy/" + video_name[:-4] + '.npy',
                M_db)


# audio_data_extraction()

#sound = np.load(r"C:/Users\dabigioi\Documents\stable-diffusion\data\GRID\S1_Audios/bbaf2n.npy")

def extract_video_frames():

    for video_name in os.listdir(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Test_Videos\21"):
        count = 0
        videodata = skvideo.io.vread(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Test_Videos\21/"+video_name)

        cap = cv2.VideoCapture(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Test_Videos\21/"+video_name)
        print(video_name)

        success, image = cap.read()
        print(success)
        current_identity = video_name[:-4]

        while success:
            cv2.imwrite(r"C:\Users\dabigioi\Documents\Diffusion_Dataset\CREMA-D\Cropped_Frames_Test\21/"+video_name[:-4]+'_'+f"{count:03d}"+'.png', image)  # save frame as PNG file
            print('written')
            success, image = cap.read()
            count += 1

#extract_video_frames()
