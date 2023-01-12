import torch.utils.data as data
from PIL import Image
import os
import mediapipe as mp
import numpy as np
from torchvision import transforms
import torch
import tqdm
import cv2
import ntpath
import time
import librosa
import skimage
import skvideo.io
import librosa.display
from matplotlib import pyplot as plt

import cv2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def contour_extractor(path_to_img):
    landmark_points_68 = [162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71,
                          63, 105, 66, 107, 336,
                          296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144,
                          362, 385, 387, 263, 373,
                          380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317,
                          14, 87]
    IMAGE_LIST = [path_to_img]
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.1) as face_mesh:
        for file in (IMAGE_LIST):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                frame_landmark_list = np.zeros((468,3))
            else:
                for face_landmarks in results.multi_face_landmarks:
                    frame_landmark_list = []
                    for i in range(0, 468):
                        pt1 = face_landmarks.landmark[i]
                        x = pt1.x
                        y = pt1.y
                        z = pt1.z
                        frame_landmark_list.append([x, y, z])
                    frame_landmark_list = np.asarray(frame_landmark_list)

            landmarks_extracted = frame_landmark_list[landmark_points_68]
            landmarks_extracted = np.asarray(landmarks_extracted)

            landmarks_extracted[:, 0] = landmarks_extracted[:, 0] * 256
            landmarks_extracted[:, 1] = landmarks_extracted[:, 1] * 256
            landmarks_extracted[:, 2] = landmarks_extracted[:, 2] * -256

            landmarks_extracted = landmarks_extracted[:, :2]

            landmark_list = []
            for items in landmarks_extracted:
                tuple = [int(items[0]), int(items[1])]
                landmark_list.append(tuple)

    return landmark_list

def face_mask_square(img_shape, landmark_list, dtype='uint8'):

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    #cv2.drawContours(mask, np.int32([landmark_list[1:16]]), -1, color=(1), thickness=cv2.FILLED)
    print(landmark_list[2])
    print(landmark_list[14])
    #print(np.int32([landmark_list[1:16]]))

    cv2.drawContours(mask, np.int32([[landmark_list[2],[landmark_list[2][0], height],landmark_list[14],[landmark_list[14][0], height], landmark_list[2],landmark_list[14],[landmark_list[2][0], height],[landmark_list[14][0], height]]]), -1, color=(1), thickness=cv2.FILLED)


    #cv2.polylines(mask, np.int32([landmark_list[1:16]]), isClosed=False, color=(0), thickness=1)  # 'Outer Mouth'
    #cv2.polylines(mask, np.int32([landmark_list[60:68]]), isClosed=False, color=(0), thickness=1)  # 'Inner Mouth'
    #cv2.line(mask, np.int32(landmark_list[48]), np.int32(landmark_list[59]), color=(0), thickness=1)
    #cv2.line(mask, np.int32(landmark_list[60]), np.int32(landmark_list[67]), color=(0), thickness=1)



    return mask

landmarks = contour_extractor(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\identity_directory_lombard/s10.jpg")
mask = face_mask_square((256,256), landmarks)

img = cv2.imread(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\identity_directory_lombard/s10.jpg")

mask_img = img*(1 - mask) + mask
plt.imshow(mask_img)


plt.show()



def pil_loader(path):
    return Image.open(path)

def rescale_frame(frame_input, percent=75):
    width = int(256)
    height = int(256)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

def rescale_videos(video_path):
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        ret, frame = cap.read()
        rescaled_frame = rescale_frame(frame)
        (h, w) = rescaled_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')


        writer = cv2.VideoWriter(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus/"+ntpath.basename(video_path)[:-4]+".mp4",
                                 fourcc, 24.0,
                                 (w, h), True)
    else:
        print("Camera is not opened")
        return

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

def audio_data_extraction():
    for video_name in os.listdir(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus_Small"):

        y, sr = librosa.load(r"C:/Users/ionut/Downloads/lombardgrid_audio/lombardgrid/audio/"+video_name[:-4]+'.wav', sr=48000)

        cap = cv2.VideoCapture(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus_Small/"+video_name)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_audio_signal_length = num_frames*2000

        if target_audio_signal_length < y.shape[0]:
            y = y[:target_audio_signal_length-1]

            padding_zeros_left = np.zeros((4000,))
            padding_zeros_right = np.zeros((4000,))
            y = np.concatenate((y, padding_zeros_right))
            y = np.concatenate((padding_zeros_left, y))

            M = librosa.feature.melspectrogram(y=y, sr=48000, n_fft=2000, hop_length=500, n_mels=256, fmax=8000)
            M_db = librosa.power_to_db(M, ref=np.max)


            np.save(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus_Small_Audio/"+video_name[:-4]+'.npy', M_db)


        elif target_audio_signal_length > y.shape[0]:

            difference = target_audio_signal_length - y.shape[0]
            zeros = np.zeros((difference-1,))
            y = np.concatenate((y, zeros))

            padding_zeros_left = np.zeros((4000,))
            padding_zeros_right = np.zeros((4000,))
            y = np.concatenate((y, padding_zeros_right))
            y = np.concatenate((padding_zeros_left, y))

            M = librosa.feature.melspectrogram(y=y, sr=48000, n_fft=2000, hop_length=500, n_mels=256, fmax=8000)
            M_db = librosa.power_to_db(M, ref=np.max)
            np.save(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus_Small_Audio/"+video_name[:-4]+'.npy', M_db)

        elif target_audio_signal_length == y.shape[0]:
            padding_zeros_left = np.zeros((4000,))
            padding_zeros_right = np.zeros((4000,))
            y = np.concatenate((y, padding_zeros_right))
            y = np.concatenate((padding_zeros_left, y))

            M = librosa.feature.melspectrogram(y=y, sr=48000, n_fft=2000, hop_length=500, n_mels=256, fmax=8000)
            M_db = librosa.power_to_db(M, ref=np.max)
            np.save(
                r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus_Small_Audio/" + video_name[
                                                                                                                                                                    :-4] + '.npy',
                M_db)

def extract_audio_and_video_frames():

    previous_identity = 'hello'
    for video_name in os.listdir(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus_Small"):

        M_db = np.load(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus_Small_Audio/"+video_name[:-4]+'.npy').T
        videodata = skvideo.io.vread(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus_Small/"+video_name)
        audio_images = []
        frame_counter = 0
        for i in range(len(videodata) - 1):
            audio_images.append(M_db[frame_counter:frame_counter + 24])
            frame_counter += 4
        audio_images = np.asarray(audio_images)
        print(audio_images.shape)
        print(M_db.shape)
        print(videodata.shape)


        cap = cv2.VideoCapture(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\Lombard_Grid_Corpus_Small/"+video_name)
        success, image = cap.read()
        current_identity = video_name.split('_')[0]
        current_count = 0
        if current_identity != previous_identity:
            count = 0
            previous_identity = current_identity
        else:
            pass
        while success:
            cv2.imwrite(r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\image_frames/"+video_name.split("_")[0]+'_'+f"{count:07d}"+'.jpg', image)  # save frame as JPEG file

            if current_count != 0:
                a = audio_images[current_count-1]
                a = np.tile(a, (10, 1))
                a = np.concatenate((a, a[:16])).T

                skimage.io.imsave(
                    r"C:\Users\ionut\Documents\My Python Projects\DiffusionDubbing\Palette-Image-to-Image-Diffusion-Models\datasets\audio_frames/" +
                    video_name.split("_")[0] + '_' + f"{count:07d}" + ".jpg", a)
            else:
                pass

            success, image = cap.read()
            count += 1
            current_count += 1