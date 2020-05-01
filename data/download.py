import os
import gdown

url = 'https://drive.google.com/uc?id=1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J'
output = 'PointDA_data.zip'
gdown.download(url, output, quiet=False)

os.system('unzip PointDA_data.zip')
os.system('rm PointDA_data.zip')