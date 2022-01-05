# Készítette: Mikula Eszter Réka
# Dátum: 2020. 11. 08.
# Script, amely előre feldolgozza a tesztalanyok képeit a pickle modul segítségével

import os
import glob
import face_recognition
import pickle
from PIL import Image, ImageDraw

def read_files(path):
    imgs = []
    for path in glob.glob(os.path.join(path, "*.*")):
        try:
           img = face_recognition.load_image_file(path)
           if img is not None:
              imgs.append(img)
        except:
            pass
    return imgs

def read_dirs(path):
    images = []
    dir_names = []

    # konyvtarak lekerese [known_faces/eszter, known_faces/tesztalany2]
    dirs = glob.glob(os.path.join(path, "*"))

    # konyvtarakon beluli fajlok beolvasasa, es a konyvtar elmentese
    for dir in dirs:
        imgs = read_files(dir)
        if len(imgs) > 0:
            images += imgs
            # a tenyleges konyvtar elmentese minden kephez [eszter,eszter,eszter,...,tesztalany2]
            dir_names += [os.path.basename(dir)] * len(imgs)
    return  images, dir_names

def encode_faces(imgs):
    lst = []
    for img in imgs:
       enc = face_recognition.face_encodings(img)
       if len(enc) > 0:
           lst += enc # lehet, hogy tobb arcot is talal egy kepen
           #lst.append(enc[0]) # ha egy kepen egy arc van
    return lst

#Képek felolvasása és kódolása
images, known_face_names = read_dirs('img/known/')
print(known_face_names)

known_face_encodings = encode_faces(images)
print("kodolas vege")

with open("dataset.pickle", 'wb') as out_file:
   pickle.dump([known_face_names, known_face_encodings], out_file)
   out_file.close()


