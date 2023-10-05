from glob import glob
from tqdm import tqdm
import os

from ClipEmbedder import ClipModel
from Database import ImageDoc, VectorDB

IMAGE_DIR = "images"
DB_DIR = "testDB"
BATCH_SIZE = 32
image_paths = glob(f"{IMAGE_DIR}/*")

model = ClipModel(half_precision=False)
db = VectorDB(
    schema=ImageDoc,
    workspace=DB_DIR,
)

if not os.path.exists(os.path.join(DB_DIR, "InMemoryExactNNIndexer[ImageDoc][ImageDocWithMatchesAndScores]", "index.bin")):
    for batch_idx in tqdm(range((len(image_paths) // BATCH_SIZE) + 1), desc="Embedding batches"):
        model = ClipModel()
        batch = image_paths[batch_idx * BATCH_SIZE:( batch_idx + 1) * BATCH_SIZE]
        image_embeddings = [model.embed_image(img) for img in batch]


        db.add_rows(
            [ImageDoc(path=path, embedding=embedding) for path, embedding in zip(batch, image_embeddings)]
        )

for (path, score) in db.query(ImageDoc(path="", embedding=model.embed_text(["dog with puppy"])), limit=5):
    print(f"{score:.3f}\t{path}")

print(
    model.text_image_probabilities(
        model.embed_image(image_paths[0]),
        model.embed_text(["golden retriever", "waterfall", "mushroom", "cat"]),
    )
)
