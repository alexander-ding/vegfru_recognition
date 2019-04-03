# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from scipy import io
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    project_dir = Path(os.getenv("project_dir")).resolve()
    input_filepath = project_dir / "data" / "raw"
    output_filepath = project_dir / "data" / "processed"
    interim_filepath = project_dir / "data" / "interim"

    logger.info('acquiring links')

    classes = []
    x_train_links = []
    y_train = []
    with open(input_filepath / "fru92_lists" / "fru_train.txt") as f:
        for line in f.readlines():
            f_path, class_no = line.split(" ")
            name = f_path.split("/")[0]
            
            if int(class_no) == len(classes):
                classes.append(name)
            
            x_train_links.append("fru92_images/" + f_path)
            y_train.append(int(class_no))

    class_base = len(classes)
    with open(input_filepath / "veg200_lists" / "veg_train.txt") as f:
        for line in f.readlines():
            f_path, class_no = line.split(" ")
            name = f_path.split("/")[0]
            
            if int(class_no)+class_base == len(classes):
                classes.append(name)
            
            x_train_links.append("veg200_images/" + f_path)
            y_train.append(int(class_no)+class_base)
    
    x_test_links = []
    y_test = []

    x_val_links = []
    y_val = []

    with open(input_filepath / "fru92_lists" / "fru_test.txt") as f:
        i = 0
        for line in f.readlines():
            if (i%8) == 0:
                f_path, class_no = line.split(" ")
                name = f_path.split("/")[0]

                x_test_links.append("fru92_images/" + f_path)
                y_test.append(int(class_no))
            i += 1

    with open(input_filepath / "veg200_lists" / "veg_test.txt") as f:
        i = 0
        for line in f.readlines():
            if (i%8) == 0:
                f_path, class_no = line.split(" ")
                name = f_path.split("/")[0]

                x_test_links.append("veg200_images/" + f_path)
                y_test.append(int(class_no)+class_base)
            i += 1
            
    with open(input_filepath / "fru92_lists" / "fru_val.txt") as f:
        for line in f.readlines():
            f_path, class_no = line.split(" ")
            name = f_path.split("/")[0]
            
            x_val_links.append("fru92_images/" + f_path)
            y_val.append(int(class_no))
            
    with open(input_filepath / "veg200_lists" / "veg_val.txt") as f:
        for line in f.readlines():
            f_path, class_no = line.split(" ")
            name = f_path.split("/")[0]
            
            x_val_links.append("veg200_images/" + f_path)
            y_val.append(int(class_no)+class_base)   

    def links_to_images(links, ys):
        y_new = []
        size = (224, 224)
        images = []
        i = 0 # counter
        for link, y in zip(links,ys):
            image = Image.open(input_filepath / link)
            image = np.array(ImageOps.fit(image, size))-128
            if image.ndim != 3 or image.shape[-1] not in (3,4):
                continue
            if image.shape[-1] != 3: # if not RGB
                images.append(image[:,:,:3])
            else:
                images.append(image)
            y_new.append(y)

            i+=1
            if (i % 1000) == 0:
                print("{} done".format(i))

        return np.array(images), y_new
        
    logger.info("processing: train")
    x_train, y_train = links_to_images(x_train_links, y_train)
    logger.info("processing: test")
    x_test, y_test = links_to_images(x_test_links, y_test)
    logger.info("processing: val")
    x_val, y_val = links_to_images(x_val_links, y_val)
    logger.info("saving data")

    np.savez(output_filepath / "vegfru_data", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_val=x_val, y_val=y_val)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
