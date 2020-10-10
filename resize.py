
from PIL import Image
img = Image.open('original.jpg')
img = img.resize((600,400),Image.BILINEAR)
img.save('test.jpg')