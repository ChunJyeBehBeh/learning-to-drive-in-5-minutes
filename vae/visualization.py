"""
Test a trained vae

python -m vae.visualization -vae logs/vae-64.pkl -f path-to-record/simulation_test/ 
"""
import argparse
import os
import sys 

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from stable_baselines.common import set_global_seeds

from .controller import VAEController

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs/recorded_data/')
    parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
    parser.add_argument('--n-samples', help='Max number of samples', type=int, default=20)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    args = parser.parse_args()

    set_global_seeds(args.seed)

    if not args.folder.endswith('/'):
        args.folder += '/'

    vae = VAEController(z_size=64)
    vae.load(args.vae_path)

    images = [im for im in os.listdir(args.folder) if im.endswith('.jpg')]
    images = np.array(images)
    n_samples = len(images)

    print("Visualization: {} Images".format(n_samples))
    print("Done Loading Parameters! Start Visualization!")

    for i in range(args.n_samples):
        #Load test image
        # image_idx = np.random.randint(n_samples)
        # image_path = args.folder + images[image_idx]
        # image = cv2.imread(image_path)
        # r = ROI
        # im = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        
        # encoded = vae.encode(image)
        # reconstructed_image = vae.decode(encoded)[0]

        # #Save Image
        # image_save = np.concatenate((image,reconstructed_image),axis=1)
        # cv2.imwrite('path-to-record/test_result/test_idx_{}.jpg'.format(image_idx),image_save)

        image_idx = np.random.randint(n_samples)
        image_path = args.folder + images[image_idx]
        image = cv2.imread(image_path)

        encoded = vae.encode_from_raw_image(image)
        reconstructed_image = vae.decode(encoded)[0]
        image_save = np.concatenate((image,reconstructed_image),axis=1)
        cv2.imwrite('path-to-record/test_result/test_idx_{}.jpg'.format(image_idx),image_save)

    
    print("Finished")
        
