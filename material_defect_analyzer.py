import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
from pathlib import Path

class MaterialDefectClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Material Surface Analyzer - Industrial Analysis Tool")
        self.root.geometry("1100x800")
        self.root.minsize(1000, 700)
        
        # Set theme colors
        self.bg_color = "#f5f7fa"
        self.accent_color = "#3498db"
        self.header_bg = "#2c3e50"
        self.header_fg = "white"
        self.result_bg = "#ecf0f1"
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Arial', 11), padding=8)
        self.style.configure('Accent.TButton', background=self.accent_color, foreground='white')
        self.style.map('Accent.TButton', background=[('active', '#2980b9')])
        
        # Model variables
        self.model = None
        self.class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
        self.img_size = 224
        
        # Class descriptions for detailed information
        self.class_descriptions = {
            'crazing': 'Fine network of surface cracks often caused by thermal stress or cooling.',
            'inclusion': 'Foreign particles embedded in the material during production.',
            'patches': 'Irregular surface areas with different texture or appearance.',
            'pitted_surface': 'Small depressions or holes on the material surface.',
            'rolled_in_scale': 'Scale particles pressed into the surface during rolling operations.',
            'scratches': 'Linear marks or grooves on the surface caused by contact with sharp objects.'
        }
        
        # Image variables
        self.selected_image_path = None
        self.processed_image = None
        self.canvas = None
        
        # Set application icon if available
        try:
            self.root.iconbitmap("icon.ico")  # You would need to create this icon file
        except:
            pass
            
        # Configure the root window's appearance
        self.root.configure(bg=self.bg_color)
        
        # Create UI
        self.create_ui()
        
        # Load model automatically (without asking user)
        self.root.after(100, self.auto_load_model)
    
    def create_ui(self):
        # Main container
        self.main_container = Frame(self.root, bg=self.bg_color)
        self.main_container.pack(fill=BOTH, expand=True, padx=15, pady=15)
        
        # Header
        header_frame = Frame(self.main_container, bg=self.header_bg, height=70)
        header_frame.pack(fill=X, pady=(0, 15))
        header_frame.pack_propagate(False)
        
        header_text = Label(
            header_frame, 
            text="MatScan: Material Surface Analyzer", 
            font=("Arial", 18, "bold"),
            bg=self.header_bg,
            fg=self.header_fg
        )
        header_text.pack(side=LEFT, padx=20, pady=15)
        
        # Status indicator in header
        self.status_frame = Frame(header_frame, bg=self.header_bg)
        self.status_frame.pack(side=RIGHT, padx=20, pady=15)
        
        self.status_indicator = Label(
            self.status_frame,
            text="◉ Ready",
            font=("Arial", 10),
            bg=self.header_bg,
            fg="#2ecc71"  # Green color for ready status
        )
        self.status_indicator.pack(side=LEFT)
        
        # Content area - split into left panel and right panel
        content_frame = Frame(self.main_container, bg=self.bg_color)
        content_frame.pack(fill=BOTH, expand=True)
        
        # Left panel - Controls and small image preview
        left_panel = Frame(content_frame, bg=self.bg_color, width=300)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Control section
        controls_frame = Frame(left_panel, bg=self.bg_color, padx=15, pady=15, relief="ridge", bd=1)
        controls_frame.pack(fill=X, pady=(0, 15))
        
        controls_label = Label(
            controls_frame,
            text="Controls",
            font=("Arial", 12, "bold"),
            bg=self.bg_color
        )
        controls_label.pack(anchor=W, pady=(0, 10))
        
        # Image selection button
        select_btn = ttk.Button(
            controls_frame, 
            text="Select Material Image",
            style='TButton',
            command=self.select_image
        )
        select_btn.pack(fill=X, pady=(0, 10))
        
        # Analyze button
        self.analyze_btn = ttk.Button(
            controls_frame, 
            text="Analyze Defect",
            style='Accent.TButton',
            command=self.classify_image,
            state=DISABLED
        )
        self.analyze_btn.pack(fill=X)
        
        # Selected file display
        self.file_frame = Frame(left_panel, bg=self.bg_color, padx=15, pady=15, relief="ridge", bd=1)
        self.file_frame.pack(fill=X, pady=(0, 15))
        
        self.file_label = Label(
            self.file_frame,
            text="No image selected",
            font=("Arial", 10),
            bg=self.bg_color,
            wraplength=250,
            justify=LEFT
        )
        self.file_label.pack(fill=X)
        
        # Information section
        info_frame = Frame(left_panel, bg=self.bg_color, padx=15, pady=15, relief="ridge", bd=1)
        info_frame.pack(fill=BOTH, expand=True)
        
        info_label = Label(
            info_frame,
            text="Information",
            font=("Arial", 12, "bold"),
            bg=self.bg_color
        )
        info_label.pack(anchor=W, pady=(0, 10))
        
        self.info_text = Label(
            info_frame,
            text="This system uses AI to detect and classify various material defects in industrial materials. Upload an image to begin analysis.",
            font=("Arial", 10),
            bg=self.bg_color,
            wraplength=250,
            justify=LEFT
        )
        self.info_text.pack(fill=X)
        
        # Right panel - Main content area with tabs
        right_panel = Frame(content_frame, bg=self.bg_color)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=BOTH, expand=True)
        
        # Image preview tab
        self.preview_tab = Frame(self.notebook, bg="white")
        self.notebook.add(self.preview_tab, text="Image Preview")
        
        # Results tab
        self.results_tab = Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.results_tab, text="Analysis Results")
        
        # Visualization tab - NEW
        self.visualization_tab = Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.visualization_tab, text="Defect Visualization")
        
        # Setup preview tab
        self.preview_container = Frame(self.preview_tab, bg="white")
        self.preview_container.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        self.preview_label = Label(
            self.preview_container, 
            text="Select an image to preview",
            font=("Arial", 14),
            bg="white"
        )
        self.preview_label.pack(fill=BOTH, expand=True)
        
        # Initialize empty results tab
        self.clear_results_tab()
        
        # Initialize empty visualization tab
        self.clear_visualization_tab()
        
        # Footer
        footer_frame = Frame(self.main_container, bg=self.bg_color, height=30)
        footer_frame.pack(fill=X, pady=(15, 0))
        
        footer_text = Label(
            footer_frame,
            text="Material Defect Classification System v1.0",
            font=("Arial", 8),
            fg="gray",
            bg=self.bg_color
        )
        footer_text.pack(side=RIGHT)
    
    def clear_results_tab(self):
        """Clear and initialize the results tab"""
        # Clear existing content
        for widget in self.results_tab.winfo_children():
            widget.destroy()
            
        # Create placeholder content
        placeholder = Label(
            self.results_tab,
            text="Analyze an image to see results",
            font=("Arial", 14),
            bg=self.bg_color
        )
        placeholder.pack(expand=True)
    
    def clear_visualization_tab(self):
        """Clear and initialize the visualization tab"""
        # Clear existing content
        for widget in self.visualization_tab.winfo_children():
            widget.destroy()
            
        # Create placeholder content
        placeholder = Label(
            self.visualization_tab,
            text="Analyze an image to see defect visualization",
            font=("Arial", 14),
            bg=self.bg_color
        )
        placeholder.pack(expand=True)
    
    def auto_load_model(self):
        """Automatically load the model without asking the user"""
        try:
            # Update status
            self.update_status("Loading model...", "orange")
            self.root.update()
            
            # First try to find model in the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_paths = [
                os.path.join(script_dir, "material_defect_classification_model.h5"),
                os.path.join(script_dir, "model", "material_defect_classification_model.h5"),
                "material_defect_classification_model.h5",
                "model/material_defect_classification_model.h5"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                # If model not found, ask user to locate it
                messagebox.showinfo("Model Not Found", "Please locate your trained model file (.h5)")
                model_path = filedialog.askopenfilename(
                    title="Select model file",
                    filetypes=[("H5 Files", "*.h5"), ("All Files", "*.*")]
                )
                
                if not model_path:
                    self.update_status("Model loading cancelled", "red")
                    return
            
            # Load the model
            self.model = tf.keras.models.load_model(model_path)
            
            # Update status indicator
            self.update_status("Model loaded successfully", "green")
            
            # Update info text
            self.info_text.config(
                text="Model loaded successfully. The system can identify six types of material defects: crazing, inclusion, patches, pitted surface, rolled-in scale and scratches. Select an image to begin."
            )
            
        except Exception as e:
            self.update_status("Error loading model", "red")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def update_status(self, message, color="green"):
        """Update the status indicator"""
        self.status_indicator.config(text=f"◉ {message}", fg=color)
    
    def select_image(self):
        """Open file dialog to select an image"""
        self.selected_image_path = filedialog.askopenfilename(
            title="Select Material Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if self.selected_image_path:
            # Update file label
            filename = os.path.basename(self.selected_image_path)
            self.file_label.config(text=f"Selected: {filename}")
            
            # Display preview
            self.display_preview()
            
            # Enable analyze button
            self.analyze_btn.config(state=NORMAL)
            
            # Switch to preview tab
            self.notebook.select(self.preview_tab)
            
            # Update status
            self.update_status("Ready to analyze")
    
    def display_preview(self):
        """Display the selected image in the preview tab"""
        # Clear existing content
        for widget in self.preview_container.winfo_children():
            widget.destroy()
        
        try:
            # Open image
            image = Image.open(self.selected_image_path)
            
            # Determine display size while maintaining aspect ratio
            width, height = image.size
            max_width = 700
            max_height = 500
            
            # Calculate new dimensions
            if width > height:
                new_width = min(width, max_width)
                new_height = int(height * (new_width / width))
            else:
                new_height = min(height, max_height)
                new_width = int(width * (new_height / height))
            
            # Resize image for display
            display_image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Create new label and display image
            preview_label = Label(self.preview_container, image=photo, bg="white")
            preview_label.image = photo  # Keep a reference
            preview_label.pack(pady=10)
            
            # Add image information
            info_text = f"Image Size: {width}x{height} pixels\nFile: {os.path.basename(self.selected_image_path)}"
            info_label = Label(
                self.preview_container,
                text=info_text,
                bg="white",
                font=("Arial", 10)
            )
            info_label.pack(pady=10)
            
        except Exception as e:
            error_label = Label(
                self.preview_container,
                text=f"Error displaying image: {str(e)}",
                bg="white",
                fg="red"
            )
            error_label.pack(expand=True)
    
    def preprocess_image(self):
        """Preprocess the image for the model"""
        img = tf.keras.preprocessing.image.load_img(
            self.selected_image_path, 
            target_size=(self.img_size, self.img_size)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        self.processed_image = np.expand_dims(img_array, axis=0)
        return self.processed_image
    
    def generate_gradcam(self, img_array, predicted_class_idx):
        """Generate a Grad-CAM heatmap for the selected class"""
        try:
            # Find the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break
            
            # If no conv layer found, use a default MobileNetV2 layer
            if not last_conv_layer:
                for layer in self.model.layers:
                    if "Conv" in layer.name:
                        last_conv_layer = layer.name
                        break
            
            # If still not found, try to get any viable layer
            if not last_conv_layer and len(self.model.layers) > 1:
                last_conv_layer = self.model.layers[-2].name
            
            # Create gradient model
            grad_model = tf.keras.models.Model(
                [self.model.inputs], 
                [self.model.get_layer(last_conv_layer).output, self.model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, predicted_class_idx]
            
            # Extract gradients
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps with gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize heatmap
            heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
            heatmap = np.uint8(255 * heatmap)
            
            # Resize heatmap to match input size
            heatmap = cv2.resize(heatmap, (self.img_size, self.img_size))
            
            return heatmap
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {str(e)}")
            return None
    
    def classify_image(self):
        """Classify the selected image and display results"""
        if not self.selected_image_path or not self.model:
            return
        
        try:
            # Update status
            self.update_status("Analyzing image...", "orange")
            self.root.update()
            
            # Preprocess image
            img_array = self.preprocess_image()
            
            # Make prediction
            predictions = self.model.predict(img_array)[0]
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = predictions[predicted_class_idx] * 100
            
            # Update results tab
            self.update_results_tab(predicted_class, confidence, predictions)
            
            # Update visualization tab
            self.update_visualization_tab(img_array, predicted_class_idx)
            
            # Switch to results tab
            self.notebook.select(self.results_tab)
            
            # Update status
            self.update_status("Analysis complete", "green")
            
        except Exception as e:
            self.update_status("Analysis failed", "red")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            print(f"Error details: {str(e)}")
    
    def update_results_tab(self, predicted_class, confidence, predictions):
        """Update the results tab with classification results"""
        # Clear results tab
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        # Create results content
        results_scroll = Frame(self.results_tab, bg=self.bg_color)
        results_scroll.pack(fill=BOTH, expand=True)
        
        # Main result section
        result_header = Label(
            results_scroll,
            text="Defect Analysis Results",
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            pady=10
        )
        result_header.pack(anchor=W, padx=20, pady=(20, 10))
        
        # Top result panel
        top_result_frame = Frame(results_scroll, bg="#e6f7e6", padx=20, pady=15, relief="ridge", bd=1)
        top_result_frame.pack(fill=X, padx=20, pady=(0, 20))
        
        result_title = Label(
            top_result_frame,
            text=f"Primary Defect: {predicted_class.upper()}",
            font=("Arial", 14, "bold"),
            bg="#e6f7e6"
        )
        result_title.pack(anchor=W)
        
        # Divider
        Frame(top_result_frame, height=1, bg="#c8e6c9", width=700).pack(fill=X, pady=10)
        
        # Description
        description = Label(
            top_result_frame,
            text=self.class_descriptions.get(predicted_class, "No description available"),
            font=("Arial", 11),
            bg="#e6f7e6",
            wraplength=700,
            justify=LEFT
        )
        description.pack(anchor=W, pady=(0, 10))
        
        # Confidence
        confidence_frame = Frame(top_result_frame, bg="#e6f7e6")
        confidence_frame.pack(fill=X, pady=(5, 0))
        
        confidence_label = Label(
            confidence_frame,
            text=f"Confidence: {confidence:.1f}%",
            font=("Arial", 11, "bold"),
            bg="#e6f7e6"
        )
        confidence_label.pack(side=LEFT)
        
        # Progress bar for confidence
        confidence_bar = ttk.Progressbar(
            top_result_frame,
            value=confidence,
            maximum=100,
            length=700,
            style="Horizontal.TProgressbar"
        )
        confidence_bar.pack(anchor=W, pady=(5, 0))
        
        # Other predictions
        other_results_frame = Frame(results_scroll, bg=self.bg_color, padx=20, pady=15)
        other_results_frame.pack(fill=X, padx=20, pady=(0, 20))
        
        other_title = Label(
            other_results_frame,
            text="Other Potential Defects",
            font=("Arial", 12, "bold"),
            bg=self.bg_color
        )
        other_title.pack(anchor=W, pady=(0, 10))
        
        # Table header
        table_frame = Frame(other_results_frame, bg=self.bg_color)
        table_frame.pack(fill=X, pady=(0, 10))
        
        # Column headers
        header_defect = Label(table_frame, text="Defect Type", font=("Arial", 10, "bold"), bg=self.bg_color, width=20, anchor=W)
        header_defect.grid(row=0, column=0, padx=5, pady=5, sticky=W)
        
        header_confidence = Label(table_frame, text="Confidence", font=("Arial", 10, "bold"), bg=self.bg_color, width=10, anchor=W)
        header_confidence.grid(row=0, column=1, padx=5, pady=5, sticky=W)
        
        # Sort predictions
        sorted_predictions = [(self.class_names[i], predictions[i] * 100) for i in range(len(predictions))]
        sorted_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Display top 3 predictions (excluding the top one if we already displayed it)
        count = 0
        for i, (defect, conf) in enumerate(sorted_predictions):
            if defect != predicted_class:  # Skip the top prediction we already showed
                defect_label = Label(table_frame, text=defect, font=("Arial", 10), bg=self.bg_color, anchor=W)
                defect_label.grid(row=i+1, column=0, padx=5, pady=2, sticky=W)
                
                conf_label = Label(table_frame, text=f"{conf:.1f}%", font=("Arial", 10), bg=self.bg_color, anchor=W)
                conf_label.grid(row=i+1, column=1, padx=5, pady=2, sticky=W)
                
                count += 1
            
            if count >= 3:
                break
                
        # Add a button to view visualization
        viz_button_frame = Frame(results_scroll, bg=self.bg_color)
        viz_button_frame.pack(fill=X, padx=20, pady=(20, 10))
        
        viz_button = ttk.Button(
            viz_button_frame,
            text="View Defect Visualization",
            style='Accent.TButton',
            command=lambda: self.notebook.select(self.visualization_tab)
        )
        viz_button.pack(side=RIGHT)
        
    def update_visualization_tab(self, img_array, predicted_class_idx):
        """Update the visualization tab with Grad-CAM heatmap"""
        # Clear visualization tab
        for widget in self.visualization_tab.winfo_children():
            widget.destroy()
            
        # Create a scrollable frame for visualization
        viz_scroll_frame = Frame(self.visualization_tab, bg=self.bg_color)
        viz_scroll_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Visualization header
        viz_header = Label(
            viz_scroll_frame,
            text="Defect Visualization",
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            pady=10
        )
        viz_header.pack(anchor=W, pady=(0, 15))
        
        # Explanation text
        explanation = Label(
            viz_scroll_frame,
            text=f"The visualization below shows where the model detected the '{self.class_names[predicted_class_idx]}' defect. Brighter areas in the heatmap indicate regions that influenced the classification decision the most.",
            font=("Arial", 11),
            bg=self.bg_color,
            wraplength=700,
            justify=LEFT
        )
        explanation.pack(anchor=W, pady=(0, 15))
        
        # Create a fixed-height frame for the matplotlib figure
        viz_display_frame = Frame(viz_scroll_frame, bg=self.bg_color, height=500)
        viz_display_frame.pack(fill=X, expand=False, pady=(0, 15))
        viz_display_frame.pack_propagate(False)  # Maintain fixed height
        
        # Create a figure with two subplots side by side
        fig = plt.figure(figsize=(10, 5))
        
        # Original image subplot
        ax1 = plt.subplot(1, 2, 1)
        img = cv2.imread(self.selected_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        ax1.imshow(img)
        ax1.set_title("Original Image", fontsize=12)
        ax1.axis('off')
        
        # Generate and display Grad-CAM
        ax2 = plt.subplot(1, 2, 2)
        heatmap = self.generate_gradcam(img_array, predicted_class_idx)
        
        if heatmap is not None:
            # Apply colormap to heatmap
            heatmap_colored = cm.jet(heatmap / 255.0)
            
            # Overlay heatmap on original image
            superimposed = img * 0.6 + heatmap_colored[:, :, :3] * 255 * 0.4
            
            ax2.imshow(superimposed.astype(np.uint8))
            ax2.set_title("Defect Heatmap (Grad-CAM)", fontsize=12)
            ax2.axis('off')
        
        # Tight layout
        plt.tight_layout(pad=3.0)
        
        # Embed matplotlib figure in tkinter
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        self.canvas = FigureCanvasTkAgg(fig, master=viz_display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Add detailed explanation
        details_frame = Frame(viz_scroll_frame, bg=self.bg_color, padx=15, pady=15, relief="ridge", bd=1)
        details_frame.pack(fill=X, pady=(10, 0))
        
        details_title = Label(
            details_frame,
            text="How to Interpret the Visualization",
            font=("Arial", 12, "bold"),
            bg=self.bg_color
        )
        details_title.pack(anchor=W, pady=(0, 10))
        
        details_text = Label(
            details_frame,
            text="• The heatmap uses color to indicate areas of interest for defect detection.\n" +
                 "• Red/yellow areas show where the model focused most when making its classification.\n" +
                 "• Blue areas had less influence on the detection decision.\n" +
                 "• This visualization helps understand why the model classified the defect as it did.",
            font=("Arial", 11),
            bg=self.bg_color,
            justify=LEFT,
            wraplength=700
        )
        details_text.pack(anchor=W)
        
        # Add a button to return to results
        button_frame = Frame(viz_scroll_frame, bg=self.bg_color)
        button_frame.pack(fill=X, pady=(15, 0))
        
        back_button = ttk.Button(
            button_frame,
            text="Back to Results",
            command=lambda: self.notebook.select(self.results_tab)
        )
        back_button.pack(side=LEFT)

# Main application
if __name__ == "__main__":
    root = Tk()
    app = MaterialDefectClassifier(root)
    root.mainloop()