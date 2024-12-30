import unittest
import torch
import tensorflow as tf
import os
import sys
from pathlib import Path
from ultralytics import YOLO

# Add parent directory to path to import convert module
sys.path.append(str(Path(__file__).parent.parent))
from convert import ModelConverter

class TestModelConverter:
    """
    Test class for ModelConverter functionality.
    Tests YOLOv8 model conversions (PyTorch and TensorFlow) to ONNX format.
    """
    
    def setup_method(self):
        """
        Set up test environment before each test method.
        Creates a ModelConverter instance and test output directory.
        Downloads YOLOv8 models if not present.
        """
        self.converter = ModelConverter()
        self.test_output_dir = "test_outputs"
        self.model_dir = "test_models"
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Download YOLOv8 model if not exists
        self.yolo_model = YOLO('yolov8n.pt')
        self.pt_path = os.path.join(self.model_dir, "yolov8n.pt")
        self.pb_path = os.path.join(self.model_dir, "yolov8n.pb")
        
        # Save models in both formats if they don't exist
        if not os.path.exists(self.pt_path):
            self.yolo_model.export(format='torchscript', save=True, save_dir=self.model_dir)
        if not os.path.exists(self.pb_path):
            self.yolo_model.export(format='pb', save=True, save_dir=self.model_dir)

    def teardown_method(self):
        """
        Clean up test environment after each test method.
        Removes test output directory and its contents.
        """
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_pytorch_yolo_conversion(self):
        """
        Test PyTorch YOLOv8 model conversion to ONNX format.
        Uses a pre-trained YOLOv8 model and verifies successful conversion.
        """
        try:
            # Load YOLOv8 PyTorch model
            model = torch.load(self.pt_path)
            # YOLOv8 expects input shape (batch_size, channels, height, width)
            dummy_input = torch.randn(1, 3, 640, 640)
            output_path = os.path.join(self.test_output_dir, "yolov8_pytorch.onnx")
            
            # Test conversion
            self.converter.convert_pytorch_to_onnx(model, dummy_input, output_path)
            assert os.path.exists(output_path), "ONNX file was not created"
        except Exception as e:
            assert False, f"PyTorch YOLOv8 conversion failed with error: {str(e)}"

    def test_tensorflow_yolo_conversion(self):
        """
        Test TensorFlow YOLOv8 model conversion to ONNX format.
        Uses a pre-trained YOLOv8 model in TensorFlow format and verifies successful conversion.
        """
        try:
            # Load YOLOv8 TensorFlow model
            model = tf.saved_model.load(self.pb_path)
            output_path = os.path.join(self.test_output_dir, "yolov8_tensorflow.onnx")
            
            # Test conversion
            self.converter.convert_tensorflow_to_onnx(model, output_path)
            assert os.path.exists(output_path), "ONNX file was not created"
        except Exception as e:
            assert False, f"TensorFlow YOLOv8 conversion failed with error: {str(e)}"

    def test_model_loading_errors(self):
        """
        Test error handling for invalid model loading attempts.
        Verifies appropriate error handling for both PyTorch and TensorFlow models.
        """
        # Test invalid PyTorch model loading
        result = self.converter.load_pytorch_model("nonexistent_model.pt")
        assert result is None, "Should return None for invalid PyTorch model"

        # Test invalid TensorFlow model loading
        result = self.converter.load_tensorflow_model("nonexistent_model.pb")
        assert result is None, "Should return None for invalid TensorFlow model"

    def test_opset_version_setting(self):
        """
        Test ModelConverter initialization with different opset versions.
        YOLOv8 typically requires opset version 11 or higher.
        """
        custom_converter = ModelConverter(opset_version=11)
        assert custom_converter.opset_version == 11, "Opset version not set correctly"

if __name__ == '__main__':
    unittest.main()