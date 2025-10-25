from utils import download_images
import pandas as pd
import os

DATA_DIR = './'
OUT_DIR = 'cache/images'
BATCH_SIZE = 10000  # adjust as needed

def main():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    all_links = pd.concat([train[['image_link']], test[['image_link']]], ignore_index=True)
    links = all_links['image_link'].fillna('').tolist()
    print(f"Total image links: {len(links)}")

    for i in range(0, len(links), BATCH_SIZE):
        batch = links[i:i+BATCH_SIZE]
        print(f"\nDownloading batch {i//BATCH_SIZE + 1} of {len(links)//BATCH_SIZE + 1} ...")
        download_images(batch, OUT_DIR)

if __name__ == "__main__":
    main()
