from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

# This is a tool that can be used by the agent to describe an image
class ImageCaptionTool(BaseTool):
    name = "Image Captioning"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')  # Convert the image to RGB

        model_name = "Salesforce/blip-image-captioning-large"  # Model name from huggingface.co/models
        device = "cuda"                                        # Device to run the model on, "cuda" for GPU, "cpu" for CPU

        processor = BlipProcessor.from_pretrained(model_name)                        # Initialize the processor
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)  # Initialize the model

        inputs = processor(image, return_tensors="pt").to(device)  # Process the image
        output = model.generate(**inputs, max_new_tokens=20)       # Generate a caption for the image

        caption = processor.decode(output[0], skip_special_tokens=True)  # Decode the caption

        # Return the caption
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async running.")


# This is a tool that can be used by the agent to detect objects in an image
class ObjectDetectionTool(BaseTool):
    name = "Object Detection"
    description = "Use this tool when given the path to an image that you would like to detect objects in. " \
                  "It will return a list of all detected objects. In the following format: " \
                  "label score"

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB') # Convert the image to RGB

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm") # Initialize the processor
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm") # Initialize the model

        inputs = processor(images=image, return_tensors="pt")  # Process the image
        outputs = model(**inputs)                              # Detect objects in the image

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]  # Post process the results

        # Return the detections in a string with labels and scores
        detections = ""
        for score, label in zip(results["scores"], results["labels"]):
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        # Return the detections
        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async running.")
