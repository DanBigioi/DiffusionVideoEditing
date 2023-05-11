import torch.utils.data as data
from PIL import Image
import os
import mediapipe as mp
import numpy as np
from torchvision import transforms
import torch
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
from .util.mask import (face_mask, bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox, face_mask_square)
from tqdm import tqdm
import random 
import PIL
import os, random


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

def make_dataset_audio(dir):
    if os.path.isfile(dir):
        audios = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        audios = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    audios.append(path)

    return audios

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

            landmarks_extracted[:, 0] = landmarks_extracted[:, 0] * 128
            landmarks_extracted[:, 1] = landmarks_extracted[:, 1] * 128
            landmarks_extracted[:, 2] = landmarks_extracted[:, 2] * -128

            landmarks_extracted = landmarks_extracted[:, :2]

            landmark_list = []
            for items in landmarks_extracted:
                tuple = [int(items[0]), int(items[1])]
                landmark_list.append(tuple)

    return landmark_list



class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[128, 128], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

class Face_Contours(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, loaderCV = cv2_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.loaderCV = loaderCV
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask(path = path)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, path):
        if self.mask_mode =='face_mask':
            print(path)
            landmark_list = contour_extractor(path)
            mask = face_mask(self.image_size, landmark_list)
        elif self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

class Video_Contouring(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, loaderCV = cv2_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.loaderCV = loaderCV
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))

        identity = os.path.basename(os.path.normpath(path))
        identity = identity.split('_')
        identity = identity[0]
        identity_path = 'datasets/identity_directory/' #path to where the identities are stored
        identity_img = self.tfs(self.loader(identity_path+identity+'.jpg'))


        mask = self.get_mask(path=path)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['identity_image'] = identity_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, path):
        if self.mask_mode =='face_mask':
            print(path)
            landmark_list = contour_extractor(path)
            mask = face_mask(self.image_size, landmark_list)
        elif self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

class VideoEditing(data.Dataset):
    def __init__(self, audio_path, img_dataset_root, mask_mode='face_mask_square', loader=pil_loader, image_size=[128,128]):

        self.imgs = make_dataset(img_dataset_root)
        self.mask_mode = mask_mode
        self.image_size = image_size
        self.audio_path = audio_path
        self.img_dataset_root = img_dataset_root

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        
        self.tfs_prevframe = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))

        mask = self.get_mask(path=path)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        frame_name = os.path.basename(os.path.normpath(path))
        frame_information = frame_name.split('_')
        frame_identity = frame_name[:-8]
        frame_number = int(frame_information[4][:-4])

        if frame_number == 0:
            previous_frame = self.tfs_prevframe(self.loader(path))
        else:      
            previous_frame_index = frame_number - 1
            previous_frame_path = self.img_dataset_root + '/' + frame_identity + '_' + f"{previous_frame_index:03d}" + '.jpg'
            previous_frame = self.tfs_prevframe(self.loader(previous_frame_path))

            
        id_frame_path = self.img_dataset_root + '/' + frame_identity + '_' + f"{0:03d}" + '.jpg'     
        id_frame = self.tfs(self.loader(id_frame_path))

        forward_window_length = 2
        backward_window_length = 2
        audio_file = np.load(self.audio_path+'/'+frame_identity+'.npy').T

        start_idx = (frame_number * 2) - backward_window_length
        end_idx = (frame_number * 2) + forward_window_length

        if start_idx < 0:
            padding = np.full((-start_idx, 256), -80)
            audio = np.concatenate((padding, audio_file[:end_idx]))

        elif end_idx > len(audio_file):
            padding = np.full((end_idx - len(audio_file), 256), -80)
            audio = np.concatenate((audio_file[start_idx:], padding))

        else:
            audio = audio_file[start_idx:end_idx]

        audio = torch.from_numpy(audio)
        audio = torch.flatten((audio))
        
        
        ret['id_frame'] = id_frame.float()
        ret['gt_image'] = img.float()
        ret['cond_image'] = cond_image.float()
        ret['mask'] = mask.float()
        ret['mask_image'] = mask_img.float()
        ret['previous_frame'] = previous_frame.float()
        ret['audio'] = audio.float()
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, path):
        landmark_list = contour_extractor(path)
        mask = face_mask_square(self.image_size, landmark_list)
        
        return torch.from_numpy(mask).permute(2,0,1)

class VideoEditing_Inference(data.Dataset):
    def __init__(self, audio_path, img_dataset_root, generated_data_root, mask_mode='face_mask_square', loader=pil_loader, image_size=[128,128]):

        self.imgs = make_dataset(img_dataset_root)
        self.mask_mode = mask_mode
        self.image_size = image_size
        self.audio_path = audio_path
        self.img_dataset_root = img_dataset_root
        self.generated_data_root = generated_data_root

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        
        self.tfs_prevframe = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        
        self.loader = loader


    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))

        mask = self.get_mask(path=path)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        frame_name = os.path.basename(os.path.normpath(path))
        frame_information = frame_name.split('_')
        frame_identity = frame_name[:-8]
        frame_number = int(frame_information[4][:-4])

        if frame_number == 0:
            previous_frame = self.tfs_prevframe(self.loader(path))
        else:
            previous_frame_index = frame_number - 1
            previous_frame_path = self.generated_data_root + '/' + 'Out_' + frame_identity + '_' + f"{previous_frame_index:03d}" + '.png'
            previous_frame = self.tfs_prevframe(self.loader(previous_frame_path))
            
            
        id_frame_path = self.generated_data_root + '/' + 'Out_' + frame_identity + '_' + f"{0:03d}" + '.png'     
        id_frame = self.tfs(self.loader(id_frame_path))

        forward_window_length = 2
        backward_window_length = 2
        audio_file = np.load(self.audio_path+'/'+frame_identity+'.npy').T

        start_idx = (frame_number * 2) - backward_window_length
        end_idx = (frame_number * 2) + forward_window_length

        if start_idx < 0:
            padding = np.full((-start_idx, 256), -80)
            audio = np.concatenate((padding, audio_file[:end_idx]))
            
        elif end_idx > len(audio_file) and start_idx > len(audio_file): 
            audio =  np.full((forward_window_length+backward_window_length, 256), -80)
        
        elif end_idx > len(audio_file):
            padding = np.full((end_idx - len(audio_file), 256), -80)
            audio = np.concatenate((audio_file[start_idx:], padding))

        else:
            audio = audio_file[start_idx:end_idx]

        audio = torch.from_numpy(audio)
        audio = torch.flatten((audio))
        
        
        ret['id_frame'] = id_frame.float()
        ret['gt_image'] = img.float()
        ret['cond_image'] = cond_image.float()
        ret['mask'] = mask.float()
        ret['mask_image'] = mask_img.float()
        ret['previous_frame'] = previous_frame.float()
        ret['audio'] = audio.float()
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, path):
        if self.mask_mode =='face_mask':
            landmark_list = contour_extractor(path)
            mask = face_mask(self.image_size, landmark_list)
        elif self.mask_mode == 'face_mask_square':
            landmark_list = contour_extractor(path)
            mask = face_mask_square(self.image_size, landmark_list)
        elif self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
        
class VideoEditing_Inference_Random_Audio(data.Dataset):
    def __init__(self, audio_path, img_dataset_root, generated_data_root, mask_mode='face_mask_square', loader=pil_loader, image_size=[128,128]):

        self.imgs = make_dataset(img_dataset_root)
        self.mask_mode = mask_mode
        self.image_size = image_size
        self.audio_path = audio_path
        self.img_dataset_root = img_dataset_root
        self.generated_data_root = generated_data_root
        self.rand_audio_name = ''

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        
        self.tfs_prevframe = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        
        self.loader = loader


    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))

        mask = self.get_mask(path=path)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        frame_name = os.path.basename(os.path.normpath(path))
        frame_information = frame_name.split('_')
        frame_identity = frame_name[:-8]
        frame_number = int(frame_information[4][:-4])

        previous_frame_index = frame_number - 1
        
        if previous_frame_index == 0:
            self.rand_audio_name = random.choice(os.listdir(self.audio_path))
            audio_file = np.load(self.audio_path+'/'+self.rand_audio_name).T
        
            previous_frame_path = self.generated_data_root + '/' + 'Out_' + frame_identity + '_' + f"{previous_frame_index:03d}" + '.png'
            previous_frame = self.tfs_prevframe(self.loader(previous_frame_path))
            
        else: 
            audio_file = np.load(self.audio_path+'/'+self.rand_audio_name).T
            previous_frame_path = self.generated_data_root + '/' + 'Out_' + frame_identity + '_' + f"{previous_frame_index:03d}" + '_' + self.rand_audio_name[:-4] + '.png'
            previous_frame = self.tfs_prevframe(self.loader(previous_frame_path))
            
        id_frame_path = self.generated_data_root + '/' + 'Out_' + frame_identity + '_' + f"{0:03d}" + '.png'     
        id_frame = self.tfs(self.loader(id_frame_path))

        forward_window_length = 2
        backward_window_length = 2
        

        start_idx = (frame_number * 2) - backward_window_length
        end_idx = (frame_number * 2) + forward_window_length

        if start_idx < 0:
            padding = np.full((-start_idx, 256), -80)
            audio = np.concatenate((padding, audio_file[:end_idx]))
            
        elif end_idx > len(audio_file) and start_idx > len(audio_file): 
            audio =  np.full((forward_window_length+backward_window_length, 256), -80)
        
        elif end_idx > len(audio_file):
            padding = np.full((end_idx - len(audio_file), 256), -80)
            audio = np.concatenate((audio_file[start_idx:], padding))

        else:
            audio = audio_file[start_idx:end_idx]

        audio = torch.from_numpy(audio)
        audio = torch.flatten((audio))
        
        
        ret['id_frame'] = id_frame.float()
        ret['gt_image'] = img.float()
        ret['cond_image'] = cond_image.float()
        ret['mask'] = mask.float()
        ret['mask_image'] = mask_img.float()
        ret['previous_frame'] = previous_frame.float()
        ret['audio'] = audio.float()
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['audio_path'] = self.rand_audio_name
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, path):
        if self.mask_mode =='face_mask':
            landmark_list = contour_extractor(path)
            mask = face_mask(self.image_size, landmark_list)
        elif self.mask_mode == 'face_mask_square':
            landmark_list = contour_extractor(path)
            mask = face_mask_square(self.image_size, landmark_list)
        elif self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

class TalkingHeadDataset(data.Dataset):
    def __init__(self, img_data_root, audio_data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, loaderCV = cv2_loader, audio_loader = pil_loader_audio):
        imgs = make_dataset(img_data_root)
        self.img_data_root = img_data_root
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        audios  = make_dataset_audio(audio_data_root)
        if data_len > 0:
            self.audios = audios[:int(data_len)]
        else:
            self.audios = audios

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.tfs_audio = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.loader = loader
        self.audio_loader = audio_loader
        self.loaderCV = loaderCV
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        img_path = self.imgs[index]
        img = self.tfs(self.loader(img_path))

        previous_frame = os.path.basename(os.path.normpath(img_path))
        previous_frame = previous_frame.split('_')
        identity = previous_frame[0]

        previous_frame = int(previous_frame[1][:-4])-1
        previous_frame_path = self.img_data_root +'/'+identity+'_'+ f"{previous_frame:07d}"+'.jpg'

        if os.path.isfile(previous_frame_path) is True:
            identity_img = self.tfs(self.loader(previous_frame_path))
        else:
            identity = os.path.basename(os.path.normpath(img_path))
            identity_matrix = identity.split('_')
            identity = identity_matrix[0]
            identity_path = 'datasets/identity_directory_lombard/'  # path to where the identities are stored
            identity_img = self.tfs(self.loader(identity_path + identity + '.jpg'))


        audio_path = self.audios[index]
        audio = self.tfs_audio(self.audio_loader(audio_path))

        mask = self.get_mask(path=img_path)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['audio'] = audio
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['identity_image'] = identity_img
        ret['mask'] = mask
        ret['path'] = img_path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, path):
        if self.mask_mode =='face_mask':
            print(path)
            landmark_list = contour_extractor(path)
            mask = face_mask(self.image_size, landmark_list)
        elif self.mask_mode == 'face_mask_square':
            print(path)
            landmark_list = contour_extractor(path)
            mask = face_mask_square(self.image_size, landmark_list)
        elif self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
       
class TalkingHeadDataset_Inference(data.Dataset):
    def __init__(self, img_data_root, audio_data_root, generated_data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, loaderCV = cv2_loader, audio_loader = pil_loader_audio):
        
        self.generated_data_root = generated_data_root
        
        imgs = make_dataset(img_data_root)
        self.img_data_root = img_data_root
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        audios  = make_dataset_audio(audio_data_root)
        if data_len > 0:
            self.audios = audios[:int(data_len)]
        else:
            self.audios = audios

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.tfs_audio = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.loader = loader
        self.audio_loader = audio_loader
        self.loaderCV = loaderCV
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        img_path = self.imgs[index]
        img = self.tfs(self.loader(img_path))

        previous_frame = os.path.basename(os.path.normpath(img_path))
        previous_frame = previous_frame.split('_')
        identity = previous_frame[0]

        previous_frame = int(previous_frame[1][:-4])-1
        previous_generated_frame_path = self.generated_data_root +'/'+'Out_'+identity+'_'+ f"{previous_frame:07d}"+'.jpg'

        if os.path.isfile(previous_generated_frame_path) is True:
            identity_img = self.tfs(self.loader(previous_generated_frame_path))
        else:
            identity = os.path.basename(os.path.normpath(img_path))
            identity_matrix = identity.split('_')
            identity = identity_matrix[0]
            identity_path = 'datasets/identity_directory/'  # path to where the identities are stored
            identity_img = self.tfs(self.loader(identity_path + identity + '.jpg'))


        audio_path = self.audios[index]
        audio = self.tfs_audio(self.audio_loader(audio_path))

        mask = self.get_mask(path=img_path)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['audio'] = audio
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['identity_image'] = identity_img
        ret['mask'] = mask
        ret['path'] = img_path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, path):
        if self.mask_mode =='face_mask':
            print(path)
            landmark_list = contour_extractor(path)
            mask = face_mask(self.image_size, landmark_list)
        elif self.mask_mode == 'face_mask_square':
            print(path)
            landmark_list = contour_extractor(path)
            mask = face_mask_square(self.image_size, landmark_list)
        elif self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


















