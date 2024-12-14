import torch
import tensorflow as tf
import tf2onnx
import onnx
import argparse
import os

class ModelConverter:
    def __init__(self, opset_version=12):
        """
        Initializes the class with the given opset version.

        :param opset_version: The opset version for ONNX (default is 12).
        """
        self.opset_version = opset_version


    def convert_pytorch_to_onnx(self, model, input_tensor, output_path):
        """
        Converts a PyTorch model to ONNX format and saves it.

        :param model: The PyTorch model (.pt file).
        :param input_tensor: The input tensor (e.g., a random tensor or actual input).
        :param output_path: The path to save the ONNX model.
        """
        try:
            print(type(model))
            model.eval()  # Set the model to evaluation mode
            # Export the PyTorch model to ONNX

            torch.onnx.export(model, 
                              input_tensor, 
                              output_path,
                              export_params=True, 
                              opset_version=self.opset_version, 
                              input_names=['input'], 
                              output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            print(f"PyTorch model successfully converted to ONNX and saved at {output_path}")
        except Exception as e:
            print(f"Error converting PyTorch model: {e}")

    def convert_tensorflow_to_onnx(self, tensorflow_model, output_path):
        """
        Converts a TensorFlow model to ONNX format and saves it.

        :param tensorflow_model: The TensorFlow model (.pb or Keras model).
        :param output_path: The path to save the ONNX model.
        """
        try:
            # Use tf2onnx to convert the TensorFlow model to ONNX
            onnx_model = tf2onnx.convert.from_keras(tensorflow_model, 
                                                     opset=self.opset_version)
            # Save the converted ONNX model
            onnx.save_model(onnx_model, output_path)
            print(f"TensorFlow model successfully converted to ONNX and saved at {output_path}")
        except Exception as e:
            print(f"Error converting TensorFlow model: {e}")

    def load_pytorch_model(self, model_path):
        """
        Loads a PyTorch model from the specified path.

        :param model_path: Path to the PyTorch model.
        :return: The loaded PyTorch model.
        """
        try:
            model = torch.load(model_path)
            print(model)
            print(f"PyTorch model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            return None

    def load_tensorflow_model(self, model_path):
        """
        Loads a TensorFlow model (Keras model or TensorFlow SavedModel) from the specified path.

        :param model_path: Path to the TensorFlow model.
        :return: The loaded TensorFlow model.
        """
        try:
            # Load Keras model or TensorFlow SavedModel
            model = tf.keras.models.load_model(model_path)
            print(f"TensorFlow model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            return None

def parse_input_size(input_size_str):
    """
    Parse the input size string (e.g., "640x640") into a tuple (640, 640)
    """
    input_size = tuple(map(int, input_size_str.split('*')))
    return input_size

def main():
    # Create an argparse object to handle command-line arguments
    parser = argparse.ArgumentParser(description="Convert PyTorch or TensorFlow model to ONNX format.")
    
    # Add arguments to the parser
    parser.add_argument('model_type', choices=['pytorch', 'tensorflow'], 
                        help="Type of model to convert (either 'pytorch' or 'tensorflow').")
    parser.add_argument('model_path', type=str, 
                        help="Path to the model file (PyTorch .pt or TensorFlow .h5).")
    parser.add_argument('output_path', type=str, 
                        help="Path to save the converted ONNX model.")
    parser.add_argument('--opset_version', type=int, default=12, 
                        help="ONNX opset version to use (default: 12).")
    parser.add_argument('--input_size', type=str, default=640*640,
                        help="The pytorch model .pt need define inputsize(default 640*640) to convert.")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Create a ModelConverter instance with the specified opset version
    converter = ModelConverter(opset_version=args.opset_version)
    
    if args.model_type == 'pytorch':
        # Load the PyTorch model
        model = converter.load_pytorch_model(args.model_path)
        if model:
            if args.input_size:
                input_size = parse_input_size(args.input_size)
                dummy_input = torch.randn(1, *input_size)
            else :
                dummy_input = torch.randn(1, 640, 640)
            converter.convert_pytorch_to_onnx(model, dummy_input, args.output_path)
    
    elif args.model_type == 'tensorflow':
        # Load the TensorFlow model
        model = converter.load_tensorflow_model(args.model_path)
        if model:
            converter.convert_tensorflow_to_onnx(model, args.output_path)

if __name__ == "__main__":
    main()
