import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.transform import resize
from sklearn.manifold import TSNE
from tqdm import tqdm


def tsne(embeddings, n_components: int = 2) -> np.ndarray:
    """
    Perform t-SNE (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction on a set of embeddings.

    Args:
        embeddings (numpy.ndarray): The input data containing the embeddings to be reduced.
        n_components (int, optional): The number of dimensions in the reduced space (default is 2).

    Returns:
        list: AN array of transformed data points in the reduced space.
    """
    # Default value is 30, sklearn raises exception when perplexity => len(embeddings), so
    # we manually set perplexity to 5 for smaller datasets
    if len(embeddings) <= 30:
        perplexity = 5
    else:
        perplexity = 30

    # Perform t-SNE dimensionality reduction on the input embeddings.
    reduced_embeddings = TSNE(n_components=n_components, perplexity=perplexity, verbose=0).fit_transform(embeddings)

    return reduced_embeddings


def rescale_image(img: np.ndarray, size: int = 256) -> np.ndarray:
    """"""
    if img.dtype != np.float32:
        img = img / 255
        
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    if width == height:
        output_size = (size, size)
    elif height > width:
        ratio = width / height
        output_size = (size, int(size * ratio))
    else:
        ratio = height / width
        output_size = (int(size * ratio), 128)
    return resize(img, output_size, preserve_range=True)
    


def image_scatterplot(
        image_paths: list[str], 
        x_coordinates: list[float], 
        y_coordinates: list[float], 
        figsize: tuple[int, int] = (16, 16)
    ) -> None:
    """
    Plots images on a scatterplot using their paths and specified x and y coordinates.

    Args:
        image_paths (list[str]): List of image file paths.
        x_coordinates (list[float]): List of x coordinates for each image.
        y_coordinates (list[float]): List of y coordinates for each image.
        figsize (tuple[int, int], optional): Figure size (width, height) in inches (default is (8, 8)).

    Returns:
        None
    """
    assert len(image_paths) == len(x_coordinates) == len(y_coordinates), "image_paths, x, and y must be same shape"
    # Create a scatterplot.
    _, ax = plt.subplots(figsize=figsize)

    # Loop through each image and its coordinates.
    for (img_path, x, y) in tqdm(zip(image_paths, x_coordinates, y_coordinates), desc="Plotting images", total=len(image_paths)):
        # Read the image using plt.imread.
        try:
            img = plt.imread(img_path)
            # img = rescale_image(img)
        except SyntaxError:
            print(img_path)

        # Create an OffsetImage for the scatterplot.
        imgbox = OffsetImage(img, zoom=0.1)  # You can adjust the 'zoom' parameter to control image size.

        # Create an AnnotationBbox to add the image to the scatterplot.
        ab = AnnotationBbox(imgbox, (x, y), frameon=False, pad=0)
        ax.add_artist(ab)

        # Set the x and y limits based on the image coordinates.
        ax.set_xlim(min(x_coordinates) - 1, max(x_coordinates) + 1)
        ax.set_ylim(min(y_coordinates) - 1, max(y_coordinates) + 1)

    # Show the plot.
    plt.axis('off')  # Turn off axis labels and ticks.
    plt.show()
