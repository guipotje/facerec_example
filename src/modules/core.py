import pickle
import os
import glob
import tqdm

from deepface.detectors import Yolo
from deepface.modules import modeling
from deepface import DeepFace

from modules.multiview import Validator

import numpy as np
import cv2

class FaceRecManager:
    def __init__(self, min_cossim = 0.75, min_real = 8, min_fake = 8):
        self.db = {}
        self.real_cnt = 0
        self.fake_cnt = 0
        self.min_real = min_real
        self.min_fake = min_fake

        self.person_name = ""
        self.ellips = None
        self.validator = Validator()

        #yolov8n face model
        self.yolo = Yolo.YoloClient()
        self.antispoof_model = modeling.build_model(model_name="Fasnet")
        self.min_cossim = min_cossim

        if os.path.exists('/db/db.pkl'):
            with open('/db/db.pkl','rb') as f:
                self.db = pickle.load(f)

        for f in tqdm.tqdm(glob.glob('/db/*/*.jpg'), desc='Loading db...'):
            if f not in self.db:
                frame = cv2.imread(f)[..., ::-1]
                #extract embeddings on detected faces
                self.db[f] = self.extract_embedding(frame)

        #Save db with possible new embeds
        with open('/db/db.pkl','wb') as f:
            pickle.dump(self.db, f)

        self.extract_embedding(np.zeros((640,640,3)), detector='skip')
        self.embeds = np.array([v for v in self.db.values()])
        self.names = [os.path.basename(os.path.dirname(v)) for v in self.db.keys()]
        print("Loaded embeddings: ", self.embeds.shape)

    def extract_embedding(self, frame, detector='yolov8'):
        embedding_obj = DeepFace.represent(
        frame,
        detector_backend = detector,
        model_name='Facenet512',
        )
        best_face = max(embedding_obj, key=lambda x: x['face_confidence'])
        embed = np.array(best_face['embedding'])
        return embed / np.linalg.norm(embed)
    
    def plot(self, frame, kpts, color=(0,255,0)):
        for kpt in kpts:
            cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, color, -1, lineType=cv2.LINE_AA) 
        return frame  
    
    def draw_box(self, frame, box):
        """
        Draws a height 128 frame by cropping original frame with box 
        and drawing it on the top left of the frame itself.
        """
        # Extract box coordinates
        x, y, w, h = box
        
        # Crop the region from the frame
        cropped_region = frame[y:y+h, x:x+w]
        
        # Calculate new width while keeping the aspect ratio
        new_height = 128
        aspect_ratio = w / h
        new_width = int(new_height * aspect_ratio)
        
        # Resize the cropped region
        resized_region = cv2.resize(cropped_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Place the resized region at the top left of the frame
        frame[0:new_height, 0:new_width] = resized_region

        return frame


    def render_text(self, frame, text, position, fill=(0, 255, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        border_thickness = thickness + 3
        
        # Draw text border
        cv2.putText(frame, text, position, font, font_scale, (0, 0, 0), border_thickness, cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(frame, text, position, font, font_scale, fill, thickness, cv2.LINE_AA)

        return frame

    def process_frame(self, frame):
        # Detect faces
        result = self.yolo.model.predict(frame, verbose=False, show=False, conf=0.65)
        plot = np.copy(frame)

        if len(result) > 0 and len(result[0].keypoints.xy) ==1  and len(result[0].boxes.xywh) == 1:
            result = result[0]
            
            plot = self.plot(plot, result.keypoints.xy.reshape(-1,2).cpu().numpy())
            box = result.boxes.xywh.view(4).cpu().long().numpy()
            x,y,w,h = box
            x-=w//2 ; y-=h//2
            box = [x,y,w,h]
            plot = self.draw_box(plot, box)

            is_real, antispoof_score = self.antispoof_model.analyze(img=frame, facial_area=box)
            if is_real and antispoof_score > 0.7:
                self.render_text(plot, "Real", (560, 40))
                self.real_cnt+=1
                self.fake_cnt=0
            elif not is_real and antispoof_score > 0.7:
                self.render_text(plot, "Fake", (560, 40), (0,0,255))
                self.real_cnt = 0
                self.fake_cnt+=1
                self.person_name = ""

            #Trigger active verification
            if self.fake_cnt > self.min_fake and self.ellips is None:
                self.ellips = self.validator.gen_random_ellipsoid(image_shape=frame.shape[:2])

            #Keep checking for the user action
            if self.ellips is not None:
                plot, mask = self.validator.draw_ellipsoid(plot, self.ellips[0], self.ellips[1])
                self.render_text(plot, "Move your head inside:", (160, 40))
                if not self.validator.check_inside(mask, box):
                    plot = self.plot(plot, result.keypoints.xy.reshape(-1,2).cpu().numpy(), color=(0,0,255))
                    self.real_cnt = 0
                    self.fake_cnt+=1
                
                #Release lock
                if self.real_cnt > self.min_real:
                    self.ellips = None

            else:
                if self.real_cnt > self.min_real and self.person_name == "":
                    embed = self.extract_embedding(frame)
                    idx, score = self.find_nearest(embed)
                    if score > self.min_cossim:
                        self.person_name =  self.names[idx]

                if self.person_name != "":
                    self.render_text(plot, self.person_name, (x,y), (3,186,252))

        return plot
    
    def find_nearest(self, query_embed):
        dist = self.embeds @ query_embed[:, None]
        cossim = np.max(dist)
        idx = np.argmax(dist)
        return idx, cossim
