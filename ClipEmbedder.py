import torch
from PIL import Image
import open_clip


class ClipModel:
    """
    A class for working with OpenAI's CLIP (Contrastive Language-Image Pre-training) model.
    
    Args:
        model_name (str): The CLIP model architecture name (default is "ViT-B-32").
        pretrained (str): The pretrained model variant (default is "laion2b_s34b_b79k").
    
    Attributes:
        model_name (str): The CLIP model architecture name.
        pretrained (str): The pretrained model variant.
        model (open_clip.models.CLIPModel): The CLIP model instance.
        preprocess (torchvision.transforms.Compose): Image preprocessing transforms.
        tokenizer (open_clip.tokenizers.Tokenizer): CLIP tokenizer for text inputs.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", half_precision: bool = True) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.half_precision = half_precision

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def embed_image(self, image_path: str) -> torch.tensor:
        """
        Embeds an image into a vector representation using the CLIP model.

        Args:
            image_path (str): The path to the image file.

        Returns:
            torch.tensor: The image embedding as a tensor.
        """
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        image_embedding = self.model.encode_image(image)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        if self.half_precision:
            image_embedding = image_embedding.half()
        return image_embedding
    
    def embed_text(self, text: list[str]) -> torch.tensor:
        """
        Embeds a list of text descriptions into vector representations using the CLIP model.

        Args:
            text (list[str]): A list of text descriptions.

        Returns:
            torch.tensor: The text embeddings as a tensor.
        """
        tokens = self.tokenizer(text)
        text_embedding = self.model.encode_text(tokens)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        if self.half_precision:
            text_embedding = text_embedding.half()
        return text_embedding
    
    @staticmethod
    def text_image_probabilities(image_embedding: torch.tensor, text_embeddings: torch.tensor) -> torch.tensor:
        """
        Computes the probabilities of text descriptions matching an image.

        Args:
            image_embedding (torch.tensor): The image embedding as a tensor.
            text_embeddings (torch.tensor): The text embeddings as a tensor.

        Returns:
            torch.tensor: The probabilities of text-image matching.
        """
        # Compute the dot product between the image and text embeddings and apply softmax
        # to obtain matching probabilities
        return (100.0 * image_embedding @ text_embeddings.T).softmax(dim=-1)


if __name__ == "__main__":
    model = ClipModel()

    image_embedding = model.embed_image("CLIP.png")
    text_embeddings = model.embed_text(["a diagram", "a dog", "a cat"])
    print(model.text_image_probabilities(image_embedding, text_embeddings))