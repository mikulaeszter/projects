# Készítette: Mikula Eszter Réka
# Dátum: 2020. 11. 08.
# A program felolvassa a "known" mappában lévő, tesztalanyokat ábrázoló képeket, a képeken
# az arcokat detektálja, kódolja, majd összehasonlítja az "unknown" mappában szereplő tesztképekkel

import os
import glob
import face_recognition
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

    # Könyvtárak lekérése
    dirs = glob.glob(os.path.join(path, "*"))

    # Könyvtárakon belüli fájlok beolvasása és a könyvtár elmentése
    for dir in dirs:
        imgs = read_files(dir)
        if len(imgs) > 0:
            images += imgs
            # A tényleges könyvtár elmentése minden képhez
            dir_names += [os.path.basename(dir)] * len(imgs)
    return images, dir_names


def encode_faces(imgs):
    lst = []
    for img in imgs:
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
            lst += enc
    return lst


# Képek felolvasása és kódolása:
images, known_face_names = read_dirs('./img/known/')
print(known_face_names)

known_face_encodings = encode_faces(images)

#Beállítja a tesztelni kívánt képeket
test_images = read_files('./img/unknown/')

#Minden egyes tesztképre megismételjük a folyamatot
for i, test_image in enumerate(test_images):

    #Megkeressük az arcokat a tesztelt képen
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    print(face_locations)
    #PIL formátumra konvertáljuk
    pil_image = Image.fromarray(test_image)

    #ImageDraw instance elkészítése
    draw = ImageDraw.Draw(pil_image)

    #For ciklussal végigmegyünk az összes talált arcon
    for(top, right, bottom, left), face_encoding in zip(face_locations,face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        print(matches)
        #Ha nem egyezik az ismert arcokkal:
        name = "Ismeretlen szemely"

        #Ha egyezést talál valamelyik arccal:
        if True in matches:
           first_match_index = matches.index(True)
           name = known_face_names[first_match_index] #Megadja a képhez tartozó nevet

        #Keretet rajzol az arc koordinátái alapján az arc köré
        draw.rectangle(((left, top),(right,bottom)), outline=(255,0,0))

        #Megrajzolja a címkét a keret alá és beilleszti a szöveget
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill = (255,0,0), outline = (255,0,0))
        draw.text((left + 6, bottom - text_height -5), name, fill = (255,255,255,255))

    del draw

    #Betölti a feldolgozott képet
    pil_image.show()

    #Elmenti a feldolgozott képet
    pil_image.save('identified'+str(i)+'.jpg')

