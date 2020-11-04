import os
import gdown

url = 'https://drive.google.com/uc?id=1dkU-8Y8K7yZaZjwelVUsxAbBf7JmOX9j'
output = 'PointSegDAdataset.rar'
gdown.download(url, output, quiet=False)

os.system('unrar x PointSegDAdataset.rar')
os.system('rm PointSegDAdataset.rar')