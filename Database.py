import os

from docarray import BaseDoc, DocList
from docarray.typing import NdArray
import numpy as np
from tqdm import tqdm
from vectordb import InMemoryExactNNVectorDB

from ClipEmbedder import ClipModel

EMBEDDING_SIZE = 512


class ImageDoc(BaseDoc):
    """
    Schema for entry in vectorDB. Represents an image path and the embedding representing the image.

    Attributes:
        path (str): The file path associated with the image.
        embedding (NdArray[512]): A 512-dimensional numpy array representing the image's embedding.
    """

    path: str = ""
    embedding: NdArray[EMBEDDING_SIZE]


class VectorDB:
    """
    A class representing a Vector Database for storing and querying image embeddings.

    Attributes:
        schema (ImageDoc): The schema for the database.
        workspace (str): The directory path where the database is stored (default is the current directory).
    """

    def __init__(self, schema: ImageDoc, workspace: str = ".", persist: bool = True) -> None:
        """
        Initializes a VectorDB instance.

        Args:
            schema (ImageDoc): The schema or document structure for the database.
            workspace (str, optional): The directory path where the database is stored (default is the current directory).
            persist (bool, optional): Save database to disk as index.bin in workspace
        """
        self.schema = schema
        self.workspace = workspace
        self.persist = persist

        # Initialize an InMemoryExactNNVectorDB instance with the specified schema and workspace.
        self.db = InMemoryExactNNVectorDB[schema](workspace=self.workspace)

    def add_rows(self, rows: list[ImageDoc]) -> None:
        """
        Adds a list of images in the database.

        Args:
            rows (list[ImageDoc]): A list of images to be indexed.
            save (bool, optional): Whether to save the database locally (default is True).
        """
        self.db.index(docs=rows)
        if self.persist:
            self.db.persist()

    def add_rows_from_image_paths(self, image_paths: str, model: ClipModel, batch_size: int=32) -> None:
        if not os.path.exists(os.path.join(self.workspace, "InMemoryExactNNIndexer[ImageDoc][ImageDocWithMatchesAndScores]", "index.bin")):
            for batch_idx in tqdm(range((len(image_paths) // batch_size) + 1), desc="Embedding batches"):
                batch = image_paths[batch_idx * batch_size:( batch_idx + 1) * batch_size]
                image_embeddings = [model.embed_image(img) for img in batch]

                self.add_rows(
                    [ImageDoc(path=path, embedding=embedding) for path, embedding in zip(batch, image_embeddings)]
                )

    def query(self, query: ImageDoc, limit: int = 99) -> list[(ImageDoc, float)]:
        """
        Queries the database with a query image and returns matching image and their scores.

        Args:
            query (ImageDoc): The query image to search for in the database.
            limit (int, optional): The maximum number of results to return (default is 99).

        Returns:
            tuple: A tuple containing two lists - the list of matching images and the list of matching scores.
            list: A list containing tuple with a path to matched image, and a score for the match.
        """
        results = self.db.search(inputs=DocList[self.schema]([query]), limit=limit)[0]

        return list(zip([m.path for m in results.matches], results.scores))

    def get_all_rows(self) -> list[(str, NdArray)]:
        """
        Retrieves all rows from the database.

        Returns:
            list: A list of tuples containing the path and NdArray for each row in the database.
        """
        query = self.schema(
            path="",
            embedding=np.zeros(shape=EMBEDDING_SIZE)
        )
        rows = self.db.search(
            inputs=DocList[self.schema]([query]),
            limit=99_999_999,  # Huge limit so we return all rows
        )[0].matches
        return [(row.path, row.embedding) for row in rows]


if __name__ == "__main__":
    db = VectorDB(schema=ImageDoc, workspace="testDB")

    # Index a list of documents with random embeddings
    test_docs = [
        ImageDoc(path=f"toy doc {i}", embedding=np.random.rand(EMBEDDING_SIZE))
        for i in range(1000)
    ]
    db.add_rows(test_docs)

    results = db.query(
        query=ImageDoc(path="query", embedding=np.random.rand(EMBEDDING_SIZE)), limit=5
    )

    print(results)
