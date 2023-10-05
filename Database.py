from docarray import BaseDoc, DocList
from docarray.typing import NdArray
import numpy as np
from vectordb import InMemoryExactNNVectorDB

class ImageDoc(BaseDoc):
    """
    Schema for entry in vectorDB. Represents an image path and the embedding representing the image.

    Attributes:
        path (str): The file path associated with the image.
        embedding (NdArray[512]): A 512-dimensional numpy array representing the image's embedding.
    """
    path: str = ''
    embedding: NdArray[512]

class VectorDB:
    """
    A class representing a Vector Database for storing and querying image embeddings.

    Attributes:
        schema (ImageDoc): The schema for the database.
        workspace (str): The directory path where the database is stored (default is the current directory).
    """

    def __init__(self, schema: ImageDoc, workspace: str = ".") -> None:
        """
        Initializes a VectorDB instance.

        Args:
            schema (ImageDoc): The schema or document structure for the database.
            workspace (str, optional): The directory path where the database is stored (default is the current directory).
        """
        self.schema = schema
        self.workspace = workspace

        # Initialize an InMemoryExactNNVectorDB instance with the specified schema and workspace.
        self.db = InMemoryExactNNVectorDB[schema](workspace=self.workspace)

    def add_rows(self, rows: list[ImageDoc], save: bool = True) -> None:
        """
        Adds a list of images in the database.

        Args:
            rows (list[ImageDoc]): A list of images to be indexed.
            save (bool, optional): Whether to save the database locally (default is True).
        """
        self.db.index(docs=rows)
        if save:
            self.db.persist()

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
    

if __name__ == "__main__":
    db = VectorDB(schema=ImageDoc, workspace="testDB")

    # Index a list of documents with random embeddings
    test_docs = [ImageDoc(path=f'toy doc {i}', embedding=np.random.rand(512)) for i in range(1000)]
    db.add_rows(test_docs)

    results = db.query(
        query=ImageDoc(path="query", embedding=np.random.rand(512)),
        limit=5
    )

    print(results)
