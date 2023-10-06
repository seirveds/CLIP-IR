from glob import glob
from tqdm import tqdm
import os
import numpy as np

from ClipEmbedder import ClipModel
from Database import ImageDoc, VectorDB

from utils import tsne, image_scatterplot

IMAGE_DIR = "images"
DB_DIR = "testDB"
BATCH_SIZE = 32
image_paths = glob(f"{IMAGE_DIR}/*")

model = ClipModel(half_precision=True)
db = VectorDB(
    schema=ImageDoc,
    workspace=DB_DIR,
)

db.add_rows_from_image_paths(image_paths, model)
#############
# Searching #
#############


for (path, score) in db.query(ImageDoc(path="", embedding=model.embed_text(["dog and puppy"])), limit=5):
    print(f"{score:.3f}\t{path}")

############################
# Zero shot classification #
############################
print(
    model.text_image_probabilities(
        model.embed_image(image_paths[0]),
        model.embed_text(["golden retriever", "waterfall", "mushroom", "cat"]),
    )
)

##############
# Clustering #
##############

paths, embeddings = zip(*db.get_all_rows())
# Transform docarray NdArrays to numpy arrays
reduced_embeddings = tsne(np.array(embeddings))
image_scatterplot(paths, reduced_embeddings[:, 0], reduced_embeddings[:, 1])
