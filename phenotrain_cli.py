import sys
import os
import argparse
import pandas as pd
import torch
from PyQt5.QtCore import pyqtSignal, QObject

# Import the deep learning modules
from deeplearning import seg_deeplab
from deeplearning import kpt_rcnn

class SignalEmitter(QObject):
    """Simple signal emitter to replace the Qt signals in the command line version"""
    signal = pyqtSignal(str)
    
    def __init__(self):
        super(SignalEmitter, self).__init__()
        self.signal.connect(self.print_signal)
    
    def emit(self, msg):
        """Print the message to console instead of emitting a Qt signal"""
        print(msg)
    
    def print_signal(self, msg):
        """Callback for the signal - not used in CLI version but kept for compatibility"""
        pass

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PhenoLearn Command Line Interface')
    
    # Create subparsers for train and predict commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command parser
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--mode', choices=['Segmentation', 'Point'], required=True,
                             help='Training mode: Segmentation or Point')
    train_parser.add_argument('--format', choices=['CSV', 'Mask'], default='CSV',
                             help='Input format for segmentation (CSV or Mask)')
    train_parser.add_argument('--csv', help='Path to CSV file with annotations')
    train_parser.add_argument('--img-dir', required=True, help='Directory containing images')
    train_parser.add_argument('--mask-dir', help='Directory containing mask images (for Mask format)')
    train_parser.add_argument('--scale', type=int, default=100, 
                             help='Image resize percentage (100 means no resize)')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--batch', type=int, default=1, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--test-percent', type=int, default=20, 
                             help='Percentage of data to use for validation')
    train_parser.add_argument('--train-level', choices=['1', '2', '3'], default='1',
                             help='Training level: 1=Minimal, 2=Intermediate, 3=Full')
    
    # Predict command parser
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--mode', choices=['Segmentation', 'Point'], required=True,
                               help='Prediction mode: Segmentation or Point')
    predict_parser.add_argument('--format', choices=['CSV', 'Mask'], default='CSV',
                               help='Output format for segmentation')
    predict_parser.add_argument('--csv', required=True, help='Path to CSV file with image names')
    predict_parser.add_argument('--img-dir', required=True, help='Directory containing images')
    predict_parser.add_argument('--model', required=True, help='Path to trained model file (.pth)')
    predict_parser.add_argument('--output-dir', required=True, help='Directory to save outputs')
    predict_parser.add_argument('--scale', type=int, default=100, 
                               help='Image resize percentage (100 means no resize)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    if args.command == 'train':
        if args.mode == 'Segmentation':
            if args.format == 'CSV' and args.csv is None:
                train_parser.error("--csv is required when format is CSV")
            elif args.format == 'Mask' and args.mask_dir is None:
                train_parser.error("--mask-dir is required when format is Mask")
        elif args.mode == 'Point' and args.csv is None:
            train_parser.error("--csv is required for Point mode")
            
    return args

def train(args):
    """Run the training process with command line arguments"""
    # Create a signal emitter for console output
    qt_signal = SignalEmitter()
    
    # Convert scale from percentage to factor
    scale = 100 / args.scale
    
    if args.mode == "Segmentation":
        if args.format == "CSV":
            seg_deeplab.train(args.img_dir, scale, args.lr, args.batch,
                        args.epochs, args.test_percent, args.train_level, 
                        qt_signal, csv_path=args.csv, mask_path=None)
        else:
            seg_deeplab.train(args.img_dir, scale, args.lr, args.batch,
                        args.epochs, args.test_percent, args.train_level, 
                        qt_signal, csv_path=None, mask_path=args.mask_dir)
    elif args.mode == "Point":
        kpt_rcnn.train(args.csv, args.img_dir, scale, args.lr, args.batch,
                    args.epochs, args.test_percent, args.train_level, qt_signal)

def predict(args):
    """Run the prediction process with command line arguments"""
    # Create a signal emitter for console output
    qt_signal = SignalEmitter()
    
    # Convert scale from percentage to factor
    scale = 100 / args.scale
    
    if args.mode == "Segmentation":
        seg_deeplab.pred(args.csv, args.img_dir, args.model, args.output_dir,
                    args.format, scale, qt_signal)
    elif args.mode == "Point":
        kpt_rcnn.pred(args.csv, args.img_dir, args.model, args.output_dir,
                    scale, qt_signal)

def main():
    """Main function to run the CLI application"""
    args = parse_arguments()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)

if __name__ == "__main__":
    main()