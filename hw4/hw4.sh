
wget http://fractal.ee.ntu.edu.tw/~derricksu/hw4_para.tar.gz -O model_para.tar.gz
tar -zxf model_para.tar.gz 
python VAE.py $1 $2
python GAN.py $2
python ACGAN.py $2
python InfoGAN.py $2
