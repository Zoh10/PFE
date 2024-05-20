from contextlib import asynccontextmanager
from fastapi import FastAPI

from ultralytics import YOLO # type: ignore
import cv2 # type: ignore
from paddleocr import PaddleOCR # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore
import math
import json
from flair.data import Sentence # type: ignore
from flair.models import SequenceTagger # type: ignore
import numpy as np # type: ignore
import re
import pathlib
import time


# Variables Globales
path = 'Dataset/image_viber_2024-02-08_10-43-10-653.png' # Va etre passe en argument

model = None 
NER_text = None
NER_tab = None
reader = None
    

def init():
    
    global model
    global NER_text
    global NER_tab
    global reader
    
    # Initialization des IAs
    model = YOLO("Models/YoloV8n.pt") # Segmentation d'image 
    NER_text = SequenceTagger.load(r'NLP_Models\SR_NER_3_fast_2\final-model.pt') # Extracteur de text
    NER_tab = SequenceTagger.load(r'NLP_Models\Table_NER_3_fast_2\final-model.pt') # Extracteur de tableau
    reader = PaddleOCR(lang='fr', show_log=False, use_angle_cls=True, enable_mkldnn=True) # , rec_model_dir='Paddle Models/ch_PP-OCRv4_rec_train') # OCR
    


def Visualize(x, y, h, w):
    
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    
    mat = np.array(Image.open(path))
    
    matrix = []
    
    top = y - h // 2
    bottom = y + h // 2
    left = x - w // 2
    right = x + w // 2
    
    for i in range(top, bottom):
        m = []
        for j in range(left, right):
            m.append(mat[i][j])
        matrix.append(m)
    return np.array(matrix)


def binarize(mat):
    copy = mat
    for i in range(len(copy)):
        for j in range(len(copy[0])):
            if copy[i][j] >= 200:
                copy[i][j] = 255
            else:
                copy[i][j] = 0
    return copy


def see(mat):
    Image.fromarray(mat).show()


def expander(Recognition, mat):
    
    copy = binarize(mat)
    rec_copy = Recognition
    
    left = int(Recognition[0][0])
    right = int(Recognition[1][0])
    line = int(Recognition[0][1] + Recognition[2][1]) // 2
    
    found = False
    
    for i in range(left, 0, -1):
        if copy[line][i] == 0:
            found = True
            new_left = i
            break
    
    if not found:
        new_left = 0
    
    found = False
    
    for i in range(right, len(copy[0])):
        if copy[line][i] == 0:
            found = True
            new_right = i
            break
    
    if not found:
        new_right = len(copy[0]) - 1

    rec_copy[0][0] = (new_left + left) // 2
    rec_copy[3][0] = (new_left + left) // 2 
    rec_copy[1][0] = (new_right + right) // 2
    rec_copy[2][0] = (new_right + right) // 2

    return rec_copy


def merger(line):
    copy = line
    restart = True 
    
    while restart:
        for i in range(1, len(copy) - 1):
            if copy[i + 1][0][0][1] - copy[i][0][2][1] <= 6:
                new_elem = [[copy[i][0][0], copy[i][0][1], copy[i + 1][0][2], copy[i + 1][0][3]], (copy[i][1][0] + " " + copy[i + 1][1][0] + " ", 0.99)]
                new_tab = []
                for j in range(0, i): 
                    new_tab.append(copy[j])
                
                new_tab.append(new_elem)
                
                for j in range(i + 2, len(copy)):
                    new_tab.append(copy[j])
                copy = new_tab
                break
        else:
            restart = False
    return copy


def best_head(heads):
    
    dict = {'DESC': [], 'QTE': [], 'PU': [], 'PT': []}  
    
    for i_2 in range(len(heads)):
        head_class = Sentence(heads[i_2][1][0])
        NER_tab.predict(head_class)
    
        key = head_class.get_labels()[0].value
        total = 0
    
        for label in head_class.get_labels():
            if label.score > total:
                total = label.score
        dict[key].append([i_2, total])

    final_heads = []
    
    for key in dict:
        index = -1
        best_score = 0
        for element in dict[key]:
            if element[1] > best_score:
                best_score = element[1]
                index = element[0]
        final_heads.append(index)
    return final_heads


def Cleaner(table):
    new_tab = []
    
    for i in range(len(table)):
        flag = True
        for j in range(len(table)):
            if i == j:
                pass
            else:
                if table[i][4] == table[j][4] and table[i][5] < table[j][5]:
                    flag = False
                    break
        if flag == True:       
            new_tab.append(table[i])
    return new_tab
    

def format_text(text):
    new_text = text
    pattern = re.compile(r'([A-Za-z])(\d+)')
    pattern2 = re.compile(r'(\d+)([A-Za-z])')
    special = [':', '_', ',']

    for element in special:
        new_text = new_text.replace(element, "")
        
    new_text = new_text.replace('@', "_")
    new_text = re.sub(pattern, r'\1 \2', new_text)
    new_text = re.sub(pattern2, r'\1 \2', new_text)

    return new_text


def process():
    
    CLASSES = ['details', 'logo', 'receiver', 'sender', 'table', 'total']
    HEADERS = [-1, -1, -1, -1] # Desc, QTE, PU, PT
    final_table = [[], [], [], []]
    final_total = [] 
    
    if np.any(np.array([model, NER_text, NER_tab, reader])) == None:
        return {"result": "error"} 
        
    toSend = {"details":{},
            'logo':[],
            'receiver':{},
            'sender':{},
            'lines':[{}],
            'total':[{}]
            }
 
    # Partie Segmentation
    results = model.predict(path, verbose=False)
    table = []

    # Cropping des parties
    for result in results:
        for box in result.boxes:
            x, y, w, h = box.xywh[0]
            table.append([x.item(), y.item(), w.item(), h.item(), int(box.cls.item()), box.conf.item()])        


    # Cleaning des doublons
    table = Cleaner(table)

    # Partie OCR
    for element in table:
        if CLASSES[element[4]] in ['table']:
            part = Visualize(element[0], element[1], element[3], element[2])
        
            result = reader.ocr(part)
        
            finish = False
        
            result[0] = sorted(result[0], key=lambda x: x[0][0][1])
        
            for line in result:
                first_line = [line[0]]
                for i in range(1, len(line)):
                    if set(range(int(line[i][0][0][1]), int(line[i][0][2][1]))) & set(range(int(first_line[0][0][0][1]), int(first_line[0][0][2][1]))) == set():
                        for i in range(len(first_line)):
                            first_line[i][0] = expander(first_line[i][0], part)
                        HEADERS = best_head(first_line)
                        break
                    first_line.append(line[i]) 
        
            for line in result:
                for i in range(len(line)):
                    for k in range(len(HEADERS)):
                            if set(range(int(line[i][0][0][0]), int(line[i][0][1][0]))) & set(range(int(first_line[HEADERS[k]][0][0][0]), int(first_line[HEADERS[k]][0][1][0]))) != set():
                                final_table[k].append(line[i])

        
            for p in range(len(final_table)):
                final_table[p] = merger(final_table[p])
        
            new_final = []
        
            for i in range(len(final_table[0])):
                new_line = [final_table[0][i][1][0]]
                for j in range(1, len(final_table)):
                    for element in final_table[j]:
                        if set(range(int(element[0][0][1]), int(element[0][2][1]))) & set(range(int(final_table[0][i][0][0][1]), int(final_table[0][i][0][2][1]))) != set():
                            new_line.append(element[1][0])
                            break
                    else:
                        new_line.append(" ")
                new_final.append(new_line)
        
            toSend["lines"] = new_final
        
        elif CLASSES[element[4]] in ['random']:
            part = Visualize(element[0], element[1], element[3], element[2])
        
            result = reader.ocr(part)
        
            header = True
            finish = False
        
            for line in result:
                first_line = [line[0]]
                search_range = range(int(first_line[0][0][0][1]) - 5, int(first_line[0][0][0][1]) + 6)
                for i in range(1, len(line)):
                    if not int(line[i][0][0][1]) in search_range and header == True:
                        header = False
                        new_line = []
                        for head in first_line:
                            final_total.append([])
                            new_line.append([head])
                
                    if header:
                        first_line.append(line[i])
                    else:
                        for k in range(len(first_line)):
                            if set(range(int(line[i][0][0][0]), int(line[i][0][1][0]))) & set(range(int(first_line[k][0][0][0]), int(first_line[k][0][1][0]))) != set():
                                new_line[k].append(line[i])
        
            maxi = 0
                
            for col in new_line:
                if len(col) >= maxi:
                    maxi = len(col)
                    best_detector = new_line.index(col)   
        
            for i in range(len(new_line[best_detector])):
                    found = False
                    for k in range(len(final_total)):
                        if k == best_detector:
                            final_total[best_detector].append(new_line[k][i][1][0])
                        else:
                            try:
                                if set(range(int(new_line[best_detector][i][0][0][1]), int(new_line[best_detector][i][0][2][1]))) & set(range(int(new_line[k][i][0][0][1]), int(new_line[k][i][0][2][1]))) != set():
                                    final_total[k].append(new_line[k][i][1][0])
                                else:
                                    final_total[k].append(" ")
                            except Exception:
                                final_total[k].append(" ")
        
            tab = np.array(final_total).T
            toSend["total"] = tab.tolist()    
                            
        elif CLASSES[element[4]] == 'logo':
            toSend['logo'] = [element[0], element[1], element[2], element[3]]
        else:

            dict = {"NAME":"", "NIF":"", "ADDRESS":"", "ART":"", "TEL":"", "RIC":"", "NIS":"", "MAIL":"", "RIB":"", "FAX":""}
        
            part = Visualize(element[0], element[1], element[3], element[2])
        
            result = reader.ocr(part)
        
            sentence = ""
        
            try:
                for line in result:
                    for extraction in line:
                        # Partie Extraction
                        if not extraction is None:
                            sentence += " " + extraction[1][0]
            except Exception:
                pass
            sentence = format_text(sentence)
            sentence = Sentence(sentence)
            NER_text.predict(sentence)
            
            for label in sentence.get_labels():
                dict[label.value] += " " + label.shortstring.replace("\"", "").rsplit("/", 1)[0]
            toSend[CLASSES[element[4]]] = dict

    return toSend


ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    init()
    
    # Load the ML model
    ml_models["process"] = process
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def predict():
    start = time.time()
    result = process()
    print(time.time() - start)
    return result