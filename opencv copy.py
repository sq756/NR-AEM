import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Tiff16Viewer:
    def __init__(self, root):
        self.root = root
        self.root.title("NR-AEM TIFF Viewer")
        self.drag_data = {"x": 0, "y": 0, "item": None}
        
        # Initialize parameters
        self.raw_image = None      # Original 16-bit data
        self.display_image = None  # 8-bit display image
        self.denoised_image = None # Denoised image
        self.mask_image = None     # Mask for selected region
        self.bg_image = None       # Background image data
        self.bg_denoised = None    # Denoised background image
        self.bg_subtracted = None  # Background subtracted image
        self.pseudo_color = None   # Pseudo-color image
        self.min_val = 0
        self.max_val = 65535
        self.kernel_size = 3      # Default denoising kernel size
        self.low_threshold = 50   # Canny edge detection low threshold
        self.high_threshold = 150 # Canny edge detection high threshold
        
        # 创建界面
        self.create_widgets()
        
        # Auto load test file
        # self.load_sample_image("square321-inter-10ccm-30s-AB-0mA-bg.tif")

    def create_widgets(self):
        # File selection buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.btn_open = tk.Button(btn_frame, text="Open 16-bit TIFF", command=self.load_image)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        self.btn_bg = tk.Button(btn_frame, text="Load Background", command=self.load_background)
        self.btn_bg.pack(side=tk.LEFT, padx=5)
        
        # Main container with scrollbar
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = tk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        
        # Configure scroll region
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        # Add scrollable frame to canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Image display area - now inside scrollable frame
        self.img_frame = tk.Frame(self.scrollable_frame)
        self.img_frame.pack()
        
        # Split into two sections for original and background images
        self.original_frame = tk.Frame(self.img_frame)
        self.original_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.background_frame = tk.Frame(self.img_frame)
        self.background_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Original image display
        self.lbl_image = tk.Label(self.original_frame)
        self.lbl_image.pack(side=tk.LEFT, padx=10)
        self.lbl_image.bind("<Button-3>", self.start_move)
        self.lbl_image.bind("<B3-Motion>", self.on_move)
        self.lbl_image.bind("<ButtonRelease-3>", self.stop_move)
        
        # Denoised image display
        self.lbl_denoised = tk.Label(self.original_frame)
        self.lbl_denoised.pack(side=tk.LEFT, padx=10)
        self.lbl_denoised.bind("<Button-3>", self.start_move)
        self.lbl_denoised.bind("<B3-Motion>", self.on_move)
        self.lbl_denoised.bind("<ButtonRelease-3>", self.stop_move)
        
        # Create third row frame for background subtracted and pseudo-color images
        self.third_frame = tk.Frame(self.img_frame)
        self.third_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Background subtracted image display
        self.lbl_bg_sub = tk.Label(self.third_frame)
        self.lbl_bg_sub.pack(side=tk.LEFT, padx=10)
        self.lbl_bg_sub.bind("<Button-3>", self.start_move)
        self.lbl_bg_sub.bind("<B3-Motion>", self.on_move)
        self.lbl_bg_sub.bind("<ButtonRelease-3>", self.stop_move)
        
        # Pseudo-color image display
        self.lbl_pseudo = tk.Label(self.third_frame)
        self.lbl_pseudo.pack(side=tk.LEFT, padx=10)
        self.lbl_pseudo.bind("<Button-3>", self.start_move)
        self.lbl_pseudo.bind("<B3-Motion>", self.on_move)
        self.lbl_pseudo.bind("<ButtonRelease-3>", self.stop_move)
        
        # Original histogram
        self.figure_orig = plt.Figure(figsize=(4,2.67))
        self.ax_orig = self.figure_orig.add_subplot(111)
        self.canvas_hist_orig = FigureCanvasTkAgg(self.figure_orig, master=self.original_frame)
        self.canvas_hist_orig.get_tk_widget().pack(side=tk.LEFT, padx=10)
        
        # Background image displays
        self.lbl_bg = tk.Label(self.background_frame)
        self.lbl_bg.pack(side=tk.LEFT, padx=10)
        self.lbl_bg.bind("<Button-3>", self.start_move)
        self.lbl_bg.bind("<B3-Motion>", self.on_move)
        self.lbl_bg.bind("<ButtonRelease-3>", self.stop_move)
        
        # Denoised background image display
        self.lbl_bg_denoised = tk.Label(self.background_frame)
        self.lbl_bg_denoised.pack(side=tk.LEFT, padx=10)
        self.lbl_bg_denoised.bind("<Button-3>", self.start_move)
        self.lbl_bg_denoised.bind("<B3-Motion>", self.on_move)
        self.lbl_bg_denoised.bind("<ButtonRelease-3>", self.stop_move)
        
        # Background histogram
        self.figure_bg = plt.Figure(figsize=(4,2.67))
        self.ax_bg = self.figure_bg.add_subplot(111)
        self.canvas_hist_bg = FigureCanvasTkAgg(self.figure_bg, master=self.background_frame)
        self.canvas_hist_bg.get_tk_widget().pack(side=tk.LEFT, padx=10)
        
        # Control panel
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10)
        
        # Intensity range control
        self.min_slider = tk.Scale(self.control_frame, from_=0, to=65535, 
                                 orient=tk.HORIZONTAL, length=400,
                                 label="Min Intensity", command=self.update_display)
        self.min_slider.pack()
        
        self.max_slider = tk.Scale(self.control_frame, from_=0, to=65535,
                                 orient=tk.HORIZONTAL, length=400,
                                 label="Max Intensity", command=self.update_display)
        self.max_slider.set(65535)
        self.max_slider.pack()
        
        # Kernel size control
        self.kernel_entry = tk.Entry(self.control_frame, width=5)
        self.kernel_entry.insert(0, "3")
        self.kernel_entry.pack(side=tk.LEFT, padx=5)
        
        # Denoising method selection
        self.denoise_method = tk.StringVar(value="mean")
        self.mean_radio = tk.Radiobutton(self.control_frame, text="Mean Filter", 
                                       variable=self.denoise_method, value="mean")
        self.mean_radio.pack(side=tk.LEFT, padx=5)
        
        self.median_radio = tk.Radiobutton(self.control_frame, text="Median Filter", 
                                        variable=self.denoise_method, value="median")
        self.median_radio.pack(side=tk.LEFT, padx=5)
        
        self.btn_denoise = tk.Button(self.control_frame, text="Apply Denoising", 
                                   command=self.apply_denoising)
        self.btn_denoise.pack(side=tk.LEFT, padx=5)
        
        # Edge detection and mask buttons
        self.btn_edge = tk.Button(self.control_frame, text="Edge Detection",
                                command=self.apply_edge_detection)
        self.btn_edge.pack(side=tk.LEFT, padx=5)
        
        self.btn_analyze = tk.Button(self.control_frame, text="Analyze Region",
                                  command=self.analyze_selected_region)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        
        # Edge detection threshold controls
        self.low_thresh_slider = tk.Scale(self.control_frame, from_=0, to=255, 
                                       orient=tk.HORIZONTAL, length=200,
                                       label="Edge Low Threshold", 
                                       command=self.update_edge_thresholds)
        self.low_thresh_slider.set(50)
        self.low_thresh_slider.pack(side=tk.LEFT, padx=5)
        
        self.high_thresh_slider = tk.Scale(self.control_frame, from_=0, to=255,
                                        orient=tk.HORIZONTAL, length=200,
                                        label="Edge High Threshold",
                                        command=self.update_edge_thresholds)
        self.high_thresh_slider.set(150)
        self.high_thresh_slider.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        self.btn_reset = tk.Button(self.control_frame, text="Reset", 
                                 command=self.reset_image)
        self.btn_reset.pack(side=tk.LEFT, padx=5)
        
        # Save buttons
        self.btn_save = tk.Button(self.control_frame, text="Save Image",
                                command=self.save_image)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        
        self.btn_save_subtracted = tk.Button(
            self.control_frame, 
            text="Save Subtracted TIF", 
            command=self.save_subtracted_tif
        )
        self.btn_save_subtracted.pack(side=tk.LEFT, padx=5)

    def load_image(self, path=None):
        if not path:
            path = filedialog.askopenfilename(filetypes=[("16-bit TIFF", "*.tif")])
        
        if path:
            try:
                # Read 16-bit TIFF
                pil_image = Image.open(path)
                self.raw_image = np.array(pil_image)
                
                # Auto set initial range
                self.min_slider.set(int(np.percentile(self.raw_image, 2)))
                self.max_slider.set(int(np.percentile(self.raw_image, 98)))
                
                self.update_display()
                self.update_histogram()
                
            except Exception as e:
                print(f"Failed to load: {str(e)}")
                
    def load_background(self, path=None):
        if not path:
            path = filedialog.askopenfilename(filetypes=[("16-bit TIFF", "*.tif")])
        
        if path:
            try:
                # Read 16-bit TIFF background
                pil_image = Image.open(path)
                self.bg_image = np.array(pil_image)
                
                # Apply denoising to background image
                from scipy.ndimage import uniform_filter, median_filter
                method = self.denoise_method.get()
                kernel_size = int(self.kernel_entry.get())
                if kernel_size % 2 == 0:
                    kernel_size += 1
                    
                if method == "mean":
                    self.bg_denoised = uniform_filter(self.bg_image, size=kernel_size)
                else:
                    self.bg_denoised = median_filter(self.bg_image, size=kernel_size)
                
                # Force update display to show denoised background
                self.update_display()
                
                print("Background loaded successfully")
                
            except Exception as e:
                print(f"Failed to load background: {str(e)}")

    def start_move(self, event):
        self.drag_data["item"] = event.widget
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        
    def on_move(self, event):
        if self.drag_data["item"]:
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            self.drag_data["item"].place(x=self.drag_data["item"].winfo_x()+dx, 
                                        y=self.drag_data["item"].winfo_y()+dy)
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            
    def stop_move(self, event):
        self.drag_data["item"] = None
        
    def create_color_legend(self):
        """Create a window showing the color gradient legend"""
        self.legend_window = tk.Toplevel(self.root)
        self.legend_window.title("Color Legend")
        
        # Create gradient image
        gradient = np.linspace(0, 1, 256).reshape(1, 256)
        gradient = np.repeat(gradient, 50, axis=0)
        
        # Apply color map
        colors = np.array([
            [0, 0, 0],        # Black
            [0, 0, 128],      # Dark blue
            [0, 0, 255],      # Blue
            [0, 255, 255],    # Cyan
            [0, 255, 0],      # Green
            [255, 255, 0],    # Yellow
            [255, 128, 0],    # Orange
            [255, 0, 0]       # Red
        ], dtype=np.uint8)
        
        indices = (gradient * (len(colors)-1)).astype(np.uint8)
        legend_img = colors[indices]
        
        # Display legend
        legend_pil = Image.fromarray(legend_img)
        legend_tk = ImageTk.PhotoImage(legend_pil)
        
        self.legend_label = tk.Label(self.legend_window, image=legend_tk)
        self.legend_label.image = legend_tk
        self.legend_label.pack()
        
        # Add min/max value labels
        self.min_label = tk.Label(self.legend_window, text="Min: 0")
        self.min_label.pack()
        
        self.max_label = tk.Label(self.legend_window, text="Max: 65535")
        self.max_label.pack()
        
    def update_legend_values(self, min_val, max_val):
        """Update the legend with current min/max values"""
        if hasattr(self, 'legend_window'):
            self.min_label.config(text=f"Min: {min_val:.1f}")
            self.max_label.config(text=f"Max: {max_val:.1f}")
    
    def add_colorbar_to_image(self, image, cmap, max_val):
        """在图像右侧添加颜色标尺"""
        try:
            # 将PIL图像转换为numpy数组
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            # 创建颜色条(宽度为20像素)
            colorbar_width = 20
            gradient = np.linspace(0, 1, h).reshape(h, 1)
            gradient = np.repeat(gradient, colorbar_width, axis=1)
            colorbar = (cmap(gradient) * 255).astype(np.uint8)
            
            # 在图像右侧添加颜色条
            combined = np.hstack((img_array, colorbar))
            
            # 添加刻度标记
            from PIL import ImageDraw, ImageFont
            pil_img = Image.fromarray(combined)
            draw = ImageDraw.Draw(pil_img)
            
            # 简单字体(如果可用可以使用更漂亮的字体)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # 添加刻度值
            positions = [0, h//4, h//2, 3*h//4, h-1]
            values = [max_val, 3*max_val//4, max_val//2, max_val//4, 0]
            
            for pos, val in zip(positions, values):
                draw.text((w + colorbar_width + 5, pos), f"{val}", fill="white", font=font)
            
            return pil_img
        except Exception as e:
            print(f"Error adding colorbar: {str(e)}")
            return image
            
    def update_display(self, *args):
            if self.raw_image is not None:
                # Get current range values
                self.min_val = self.min_slider.get()
                self.max_val = self.max_slider.get()
                
                # If no display image exists, create from raw image
                if not hasattr(self, 'display_image') or self.display_image is None:
                    self.display_image = Image.fromarray((self.raw_image/256).astype(np.uint8))
                
                # Linear mapping to 0-255
                scaled = np.clip((self.raw_image - self.min_val) * (255.0 / (self.max_val - self.min_val)), 0, 255)
                self.display_image = Image.fromarray(scaled.astype(np.uint8))
                
                # Perform background subtraction if background is loaded
                if self.bg_image is not None and self.bg_denoised is not None:
                    # Verify background subtraction calculation
                    bg_sub = self.raw_image.astype(np.float32) - self.bg_denoised.astype(np.float32)
                    print(f"Raw subtraction range: {np.min(bg_sub)} to {np.max(bg_sub)}")
                    
                    # 偏移使最小值为0
                    bg_sub = bg_sub - np.min(bg_sub)
                    self.bg_subtracted = np.clip(bg_sub, 0, 65535).astype(np.uint16)
                    
                    # Create smooth gradient colormap from black to red
                    from matplotlib.colors import LinearSegmentedColormap
                    colors = ["black", "blue", "darkblue", "cyan", "green", "yellow", "orange", "red"]
                    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
                    
                    # Create pseudo-color image with fixed max 400
                    fixed_max = 400
                    norm_data = np.clip(self.bg_subtracted, 0, fixed_max) / fixed_max
                    pseudo_pil = Image.fromarray((cmap(norm_data) * 255).astype(np.uint8))
                    self.pseudo_color = self.add_colorbar_to_image(pseudo_pil, cmap, fixed_max)
                    
                    # Ensure legend window stays open
                    if not hasattr(self, 'legend_window') or not self.legend_window.winfo_exists():
                        self.create_color_legend()
                        
                        # Update legend with current min/max values
                        if hasattr(self, 'legend_window'):
                            self.update_legend_values(np.min(self.bg_subtracted), np.max(self.bg_subtracted))
                    
                    # 改进显示范围设置
                    bg_min = np.percentile(self.bg_subtracted, 2)
                    bg_max = np.percentile(self.bg_subtracted, 98)
                    
                    # 确保有足够的动态范围
                    if bg_max - bg_min < 100:  # 如果范围太小
                        bg_max = bg_min + 100  # 设置最小范围
                    
                    # Display background subtracted image with proper scaling
                    if bg_max > bg_min:
                        bg_sub_scaled = np.clip((self.bg_subtracted - bg_min) * (255.0 / (bg_max - bg_min)), 0, 255)
                    else:
                        bg_sub_scaled = np.zeros_like(self.bg_subtracted, dtype=np.uint8)
                    
                    print(f"Display range: {bg_min} to {bg_max}, scaled range: {np.min(bg_sub_scaled)} to {np.max(bg_sub_scaled)}")
                    bg_sub_display = Image.fromarray(bg_sub_scaled.astype(np.uint8))
                    resized_bg_sub = bg_sub_display.resize((341, 341))
                    imgtk_bg_sub = ImageTk.PhotoImage(resized_bg_sub)
                    self.lbl_bg_sub.config(image=imgtk_bg_sub)
                    self.lbl_bg_sub.image = imgtk_bg_sub
                    
                    # Display pseudo-color image
                    resized_pseudo = self.pseudo_color.resize((341, 341))
                    imgtk_pseudo = ImageTk.PhotoImage(resized_pseudo)
                    self.lbl_pseudo.config(image=imgtk_pseudo)
                    self.lbl_pseudo.image = imgtk_pseudo
                
                # Adjust display size
                display_size = (341, 341)  # 缩放到适合显示的尺寸
                resized_image = self.display_image.resize(display_size)
                
                # Convert to Tkinter displayable image
                imgtk = ImageTk.PhotoImage(resized_image)
                
                # Update display
                self.lbl_image.config(image=imgtk)
                self.lbl_image.image = imgtk
                
                # Update denoised image display if exists
                if self.denoised_image is not None:
                    scaled_denoised = np.clip((self.denoised_image - self.min_val) * (255.0 / (self.max_val - self.min_val)), 0, 255)
                    denoised_display = Image.fromarray(scaled_denoised.astype(np.uint8))
                    resized_denoised = denoised_display.resize(display_size)
                    imgtk_denoised = ImageTk.PhotoImage(resized_denoised)
                    self.lbl_denoised.config(image=imgtk_denoised)
                    self.lbl_denoised.image = imgtk_denoised
                    
                    # Update original histogram
                    self.update_histogram()
                    
                # Update background image display if exists
                if self.bg_image is not None:
                    # Display original background image
                    scaled_bg = np.clip((self.bg_image - self.min_val) * (255.0 / (self.max_val - self.min_val)), 0, 255)
                    bg_display = Image.fromarray(scaled_bg.astype(np.uint8))
                    resized_bg = bg_display.resize(display_size)
                    imgtk_bg = ImageTk.PhotoImage(resized_bg)
                    
                    self.lbl_bg.config(image=imgtk_bg)
                    self.lbl_bg.image = imgtk_bg
                    
                    # Display denoised background image if exists
                    if self.bg_denoised is not None:
                        scaled_bg_denoised = np.clip((self.bg_denoised - self.min_val) * (255.0 / (self.max_val - self.min_val)), 0, 255)
                        bg_denoised_display = Image.fromarray(scaled_bg_denoised.astype(np.uint8))
                        resized_bg_denoised = bg_denoised_display.resize(display_size)
                        imgtk_bg_denoised = ImageTk.PhotoImage(resized_bg_denoised)
                        
                        self.lbl_bg_denoised.config(image=imgtk_bg_denoised)
                        self.lbl_bg_denoised.image = imgtk_bg_denoised
                    
                    # Update background histogram
                    self.update_bg_histogram()
    
    def apply_denoising(self):
        if self.raw_image is not None:
            try:
                kernel_size = int(self.kernel_entry.get())
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Ensure odd kernel size
                
                # Apply selected denoising method
                from scipy.ndimage import uniform_filter, median_filter
                method = self.denoise_method.get()
                
                if method == "mean":
                    self.denoised_image = uniform_filter(self.raw_image, size=kernel_size)
                    if self.bg_image is not None:
                        self.bg_denoised = uniform_filter(self.bg_image, size=kernel_size)
                else:
                    self.denoised_image = median_filter(self.raw_image, size=kernel_size)
                    if self.bg_image is not None:
                        self.bg_denoised = median_filter(self.bg_image, size=kernel_size)
                
                self.update_display()
                self.update_histogram()
                self.update_bg_histogram()
            except Exception as e:
                print(f"Denoising failed: {str(e)}")

    def apply_edge_detection(self):
        if self.raw_image is not None:
            # Create edge detection window
            edge_window = tk.Toplevel(self.root)
            edge_window.title("Edge Detection & Mask Tool")
            
            # Create EdgeDetectionTool class instance
            edge_tool = EdgeDetectionTool(edge_window, self.raw_image, 
                                         self.low_threshold, self.high_threshold)
            
            # Wait for window to close
            edge_window.wait_window()
            
            # Get results if available
            if edge_tool.mask_image is not None:
                self.mask_image = edge_tool.mask_image
                self.update_display()
                self.update_histogram()
                
                # Show mask in denoised image display
                if self.denoised_image is not None:
                    masked_img = self.denoised_image.copy()
                    masked_img[self.mask_image == 0] = 0
                    self.denoised_image = masked_img
                    self.update_display()
                
                # Show statistics
                self.analyze_selected_region()

    def update_edge_thresholds(self, *args):
        """Update edge detection thresholds from sliders"""
        self.low_threshold = self.low_thresh_slider.get()
        self.high_threshold = self.high_thresh_slider.get()

    def analyze_selected_region(self):
        if self.raw_image is not None and self.mask_image is not None:
            masked_data = self.raw_image[self.mask_image == 1]
            
            self.ax.clear()
            self.ax.hist(masked_data, bins=256, range=(0,65535), 
                        log=True, histtype='stepfilled')
            
            stats_text = f"""Selected Region Statistics:
Mean: {np.mean(masked_data):.1f}
Std Dev: {np.std(masked_data):.1f}
Max: {np.max(masked_data)}
Min: {np.min(masked_data)}
Pixel Count: {len(masked_data)}"""
            
            self.ax.text(0.98, 0.98, stats_text,
                       transform=self.ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8))
            
            self.ax.set_title("Selected Region Histogram (Log Scale)")
            self.ax.set_xlabel("Intensity Value")
            self.ax.set_ylabel("Pixel Count (log)")
            self.canvas.draw()

    def update_histogram(self):
        if self.raw_image is not None:
            self.ax_orig.clear()
            
            # Display histogram with log scale
            hist_data = self.raw_image.flatten()
            self.ax_orig.hist(hist_data, bins=256, range=(0,65535), 
                        log=True, histtype='stepfilled')
            
            # Display statistics
            stats_text = f"""Original Image Statistics:
Mean: {np.mean(hist_data):.1f}
Std Dev: {np.std(hist_data):.1f}
Max: {np.max(hist_data)}
Min: {np.min(hist_data)}"""
            
            self.ax_orig.text(0.98, 0.98, stats_text,
                       transform=self.ax_orig.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8))
            
            self.ax_orig.set_title("Original Image Histogram (Log Scale)")
            self.ax_orig.set_xlabel("Intensity Value")
            self.ax_orig.set_ylabel("Pixel Count (log)")
            self.canvas_hist_orig.draw()
            
    def update_bg_histogram(self):
        if self.bg_image is not None:
            self.ax_bg.clear()
            bg_data = self.bg_denoised.flatten() if self.bg_denoised is not None else self.bg_image.flatten()
            
            self.ax_bg.hist(bg_data, bins=256, range=(0,65535), 
                        log=True, histtype='stepfilled', color='blue', alpha=0.5)
            
            # Display statistics
            stats_text = f"""Background Statistics:
Mean: {np.mean(bg_data):.1f}
Std Dev: {np.std(bg_data):.1f}
Max: {np.max(bg_data)}
Min: {np.min(bg_data)}"""
            
            self.ax_bg.text(0.98, 0.98, stats_text,
                       transform=self.ax_bg.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8))
            
            self.ax_bg.set_title("Background Histogram (Log Scale)")
            self.ax_bg.set_xlabel("Intensity Value")
            self.ax_bg.set_ylabel("Pixel Count (log)")
            self.canvas_hist_bg.draw()
    
    def save_image(self):
        """Save current display as high-quality image or full window screenshot"""
        if self.raw_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Save entire window as screenshot (including title and all images)
                import pyautogui
                x = self.root.winfo_rootx()
                y = self.root.winfo_rooty()
                width = self.root.winfo_width()
                height = self.root.winfo_height()
                screenshot = pyautogui.screenshot(region=(x, y, width, height))
                
                # Save with high quality
                if file_path.lower().endswith('.jpg'):
                    screenshot.save(file_path, quality=95)
                else:
                    screenshot.save(file_path)
                
                print(f"Saved screenshot to {file_path}")
            except Exception as e:
                print(f"Failed to save image: {str(e)}")
                
    def save_subtracted_tif(self):
        """保存背景减法后的灰度图像为2048x2048的TIFF文件"""
        if self.bg_subtracted is None:
            tk.messagebox.showerror("Error", "No subtracted image available!")
            return
            
        # 弹出文件保存对话框
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")],
            title="Save Subtracted Image As"
        )
        
        if not file_path:  # 用户取消了保存
            return
        
        try:
            # 确保图像是16位灰度
            img_to_save = self.bg_subtracted.astype(np.uint16)
            
            # 如果需要调整大小到2048x2048
            if img_to_save.shape != (2048, 2048):
                from skimage.transform import resize
                img_to_save = resize(
                    img_to_save, 
                    (2048, 2048), 
                    order=1,  # 1表示双线性插值
                    preserve_range=True,
                    anti_aliasing=True
                ).astype(np.uint16)
            
            # 使用PIL保存为16位TIFF
            tif_image = Image.fromarray(img_to_save)
            tif_image.save(file_path, format="TIFF")
            
            tk.messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
    
    def reset_image(self):
        """Reset image to original loaded state"""
        if self.raw_image is not None:
            # Reset all parameters to initial values
            self.min_slider.set(int(np.percentile(self.raw_image, 2)))
            self.max_slider.set(int(np.percentile(self.raw_image, 98)))
            self.denoised_image = None
            self.mask_image = None
            self.kernel_entry.delete(0, tk.END)
            self.kernel_entry.insert(0, "3")
            self.denoise_method.set("mean")
            self.low_thresh_slider.set(50)
            self.high_thresh_slider.set(150)
            
            # Update display
            self.update_display()
            self.update_histogram()

class FreeSelectionTool:
    def __init__(self, root, image_data):
        self.root = root
        self.root.title("Free Selection Tool")
        
        # 原始图像数据
        self.raw_image = image_data
        self.mask = np.zeros_like(image_data, dtype=np.uint8)
        
        # 选择的点
        self.points = []
        self.drawing = False
        self.draw_mode = "free"  # free/line
        self.brush_size = 5
        self.stats_window = None
        
        # 创建UI
        self.create_widgets()
        
        # 初始化显示
        self.update_display()
    
    def create_widgets(self):
        # 主画布
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 绘制模式选择
        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value="free")
        tk.Radiobutton(mode_frame, text="Free Draw", 
                      variable=self.mode_var, value="free").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Line Draw", 
                      variable=self.mode_var, value="line").pack(side=tk.LEFT)
        
        # 画笔大小控制
        size_frame = tk.Frame(control_frame)
        size_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(size_frame, text="Brush Size:").pack(side=tk.LEFT)
        self.brush_slider = tk.Scale(size_frame, from_=1, to=20, 
                                   orient=tk.HORIZONTAL, length=100,
                                   command=self.update_brush_size)
        self.brush_slider.set(5)
        self.brush_slider.pack(side=tk.LEFT)
        
        # 操作按钮
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT, padx=5)
        
        tk.Button(btn_frame, text="Clear", 
                 command=self.clear_selection).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Apply", 
                 command=self.apply_selection).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Cancel", 
                 command=self.root.destroy).pack(side=tk.LEFT)
        
        # 绑定鼠标事件
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
    
    def update_display(self):
        self.ax.clear()
        
        # 显示原始图像
        self.ax.imshow(self.raw_image, cmap='gray', vmin=0, vmax=65535)
        
        # 显示当前选区
        if len(self.points) > 1:
            x, y = zip(*self.points)
            if self.draw_mode == "free":
                self.ax.plot(x, y, 'r-', linewidth=self.brush_size)
            else:
                if len(self.points) >= 2:
                    self.ax.plot([x[0], x[-1]], [y[0], y[-1]], 'r-', linewidth=self.brush_size)
        
        # 显示已应用的掩膜
        if np.any(self.mask):
            masked = np.ma.masked_where(self.mask == 0, self.mask)
            self.ax.imshow(masked, cmap='autumn', alpha=0.5)
        
        self.ax.set_title("Free Selection Tool")
        self.canvas.draw()
        
    def update_brush_size(self, val):
        self.brush_size = int(val)
        self.update_display()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
            
        self.drawing = True
        self.draw_mode = self.mode_var.get()
        self.points.append((event.xdata, event.ydata))
        self.update_display()
    
    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
            
        if self.draw_mode == "free":
            self.points.append((event.xdata, event.ydata))
            self.update_display()
    
    def on_release(self, event):
        self.drawing = False
        if len(self.points) > 2 and self.draw_mode == "line":
            self.points.append(self.points[0])  # 闭合路径
        self.update_display()
    
    def clear_selection(self):
        self.points = []
        self.mask.fill(0)
        self.update_display()
    
    def apply_selection(self):
        if len(self.points) > 2:
            # 创建路径
            path = Path(self.points)
            
            # 创建网格
            h, w = self.raw_image.shape
            y, x = np.mgrid[:h, :w]
            coords = np.vstack((x.ravel(), y.ravel())).T
            
            # 创建掩膜
            mask = path.contains_points(coords).reshape((h, w))
            self.mask = mask.astype(np.uint8)
            
            # 显示结果
            self.update_display()
            
            # 关闭窗口
            self.root.destroy()

import cv2

class EdgeDetectionTool:
    def __init__(self, root, raw_image, low_threshold, high_threshold):
        self.root = root
        self.raw_image = raw_image
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.mask_image = None
        self.denoised_image = None
        self.min_slider = None
        self.max_slider = None
        self.drawing = False
        self.last_point = None
        self.masks = {}  # 存储多个mask
        self.draw_mode = "free"  # free/line
        self.brush_size = 5
        self.stats_window = None
        
        # Create display image (8-bit)
        self.display_image = Image.fromarray((raw_image/256).astype(np.uint8))
        
        # Setup UI
        self.create_widgets()
        
        # Initialize sliders after widgets are created
        # These are now properly initialized in create_widgets()
    
    def create_widgets(self):
        # Main image display
        self.img_frame = tk.Frame(self.root)
        self.img_frame.pack(pady=10)
        
        # Original image display
        self.lbl_image = tk.Label(self.img_frame)
        self.lbl_image.pack(side=tk.LEFT, padx=10)
        self.lbl_image.bind("<Button-1>", self.start_drawing)
        self.lbl_image.bind("<B1-Motion>", self.draw)
        self.lbl_image.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Initialize mask image
        self.mask_image = np.zeros_like(self.raw_image, dtype=np.uint8)
        self.drawing = False
        self.last_point = None
        
        # Denoised image display
        self.lbl_denoised = tk.Label(self.img_frame)
        self.lbl_denoised.pack(side=tk.LEFT, padx=10)
        
        # Initialize display image if not exists
        if not hasattr(self, 'display_image') or self.display_image is None:
            self.display_image = Image.fromarray((self.raw_image/256).astype(np.uint8))
            
        # Initialize sliders before updating display
        self.min_slider = tk.Scale(self.root, from_=0, to=65535, 
                                 orient=tk.HORIZONTAL, length=200,
                                 label="Min Intensity")
        self.max_slider = tk.Scale(self.root, from_=0, to=65535,
                                  orient=tk.HORIZONTAL, length=200,
                                  label="Max Intensity")
        self.max_slider.set(65535)
        
        # Update initial display after sliders are initialized
        self.update_display()
        
        # Control panel
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10)
        
        # Edge detection methods
        self.method_var = tk.StringVar(value="canny")
        
        tk.Radiobutton(self.control_frame, text="Canny Edge", 
                      variable=self.method_var, value="canny").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.control_frame, text="Sobel Edge", 
                      variable=self.method_var, value="sobel").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.control_frame, text="Manual Draw", 
                      variable=self.method_var, value="manual").pack(side=tk.LEFT, padx=5)
                      
        # Drawing controls
        self.draw_mode_var = tk.StringVar(value="free")
        tk.Radiobutton(self.control_frame, text="Free Draw", 
                      variable=self.draw_mode_var, value="free").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.control_frame, text="Line Draw", 
                      variable=self.draw_mode_var, value="line").pack(side=tk.LEFT, padx=5)
                      
        # Brush size control
        tk.Label(self.control_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_slider = tk.Scale(self.control_frame, from_=1, to=20, 
                                   orient=tk.HORIZONTAL, length=100,
                                   command=self.update_brush_size)
        self.brush_slider.set(5)
        self.brush_slider.pack(side=tk.LEFT, padx=5)
        
        # Intensity range controls
        self.min_slider = tk.Scale(self.control_frame, from_=0, to=65535, 
                                 orient=tk.HORIZONTAL, length=200,
                                 label="Min Intensity", command=self.update_display)
        self.min_slider.pack(side=tk.LEFT, padx=5)
        
        self.max_slider = tk.Scale(self.control_frame, from_=0, to=65535,
                                  orient=tk.HORIZONTAL, length=200,
                                  label="Max Intensity", command=self.update_display)
        self.max_slider.set(65535)
        self.max_slider.pack(side=tk.LEFT, padx=5)
        
        # Threshold controls
        self.low_slider = tk.Scale(self.control_frame, from_=0, to=255, 
                                 orient=tk.HORIZONTAL, label="Low Threshold",
                                 command=self.update_thresholds)
        self.low_slider.set(self.low_threshold)
        self.low_slider.pack(side=tk.LEFT, padx=5)
        
        self.high_slider = tk.Scale(self.control_frame, from_=0, to=255,
                                  orient=tk.HORIZONTAL, label="High Threshold",
                                  command=self.update_thresholds)
        self.high_slider.set(self.high_threshold)
        self.high_slider.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        tk.Button(self.control_frame, text="Detect Edges", 
                 command=self.detect_edges).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Apply Mask", 
                 command=self.apply_mask).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Cancel", 
                 command=self.root.destroy).pack(side=tk.LEFT, padx=5)
    
    def update_display(self):
        """Update image display"""
        display_size = (512, 512)
        
        # Get current range values
        min_val = self.min_slider.get()
        max_val = self.max_slider.get()
        
        # If no display image exists, create from raw image
        if not hasattr(self, 'display_image') or self.display_image is None:
            self.display_image = Image.fromarray((self.raw_image/256).astype(np.uint8))
        
        # For edge detection results (RGB images), skip the scaling
        if isinstance(self.display_image, np.ndarray) and len(self.display_image.shape) == 3:
            resized_image = Image.fromarray(self.display_image).resize(display_size)
        else:
            # Linear mapping to 0-255 for grayscale images
            scaled = np.clip((self.raw_image - min_val) * (255.0 / (max_val - min_val)), 0, 255)
            self.display_image = Image.fromarray(scaled.astype(np.uint8))
            resized_image = self.display_image.resize(display_size)
            
        imgtk = ImageTk.PhotoImage(resized_image)
        self.lbl_image.config(image=imgtk)
        self.lbl_image.image = imgtk
        
        # Update denoised image display if exists
        if self.denoised_image is not None:
            scaled_denoised = np.clip((self.denoised_image - min_val) * (255.0 / (max_val - min_val)), 0, 255)
            denoised_display = Image.fromarray(scaled_denoised.astype(np.uint8))
            resized_denoised = denoised_display.resize(display_size)
            imgtk_denoised = ImageTk.PhotoImage(resized_denoised)
            self.lbl_denoised.config(image=imgtk_denoised)
            self.lbl_denoised.image = imgtk_denoised
    
    def update_thresholds(self, *args):
        """Update edge detection thresholds"""
        self.low_threshold = self.low_slider.get()
        self.high_threshold = self.high_slider.get()
    
    def detect_edges(self):
        """Apply selected edge detection method"""
        try:
            import cv2
            method = self.method_var.get()
            
            # Use denoised image if available
            image_to_process = self.denoised_image if self.denoised_image is not None else self.raw_image
            
            if method == "canny":
                edges = cv2.Canny(image_to_process.astype(np.uint8), 
                                 self.low_threshold, self.high_threshold)
                # Create RGBA image with transparent background and red edges
                colored_edges = np.zeros((edges.shape[0], edges.shape[1], 4), dtype=np.uint8)
                colored_edges[edges > 0] = [255, 0, 0, 255]  # Set edges to red with full opacity
                colored_edges[edges == 0] = [0, 0, 0, 0]      # Set background to transparent
                self.display_image = Image.fromarray(colored_edges, 'RGBA')
            elif method == "sobel":
                sobelx = cv2.Sobel(image_to_process, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(image_to_process, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sobelx**2 + sobely**2)
                edges = (edges * (255.0 / edges.max())).astype(np.uint8)
                # Create RGBA image with transparent background and red edges
                colored_edges = np.zeros((edges.shape[0], edges.shape[1], 4), dtype=np.uint8)
                colored_edges[edges > 128] = [255, 0, 0, 255]  # Set strong edges to red with full opacity
                colored_edges[edges <= 128] = [0, 0, 0, 0]      # Set background to transparent
                self.display_image = Image.fromarray(colored_edges, 'RGBA')
            
            self.update_display()
        except Exception as e:
            print(f"Edge detection failed: {str(e)}")
    
    def start_drawing(self, event):
        """Start drawing mask"""
        self.drawing = True
        self.last_point = (event.x, event.y)
        
        # Initialize mask if not exists
        if self.mask_image is None:
            h, w = self.raw_image.shape
            self.mask_image = np.zeros((h, w), dtype=np.uint8)
    
    def update_brush_size(self, *args):
        """Update brush size from slider"""
        self.brush_size = self.brush_slider.get()
        
    def draw(self, event):
        """Draw mask while mouse moving"""
        if self.drawing:
            try:
                import cv2
                x, y = event.x, event.y
                
                # Scale coordinates to match original image size
                scale_x = self.raw_image.shape[1] / self.lbl_image.winfo_width()
                scale_y = self.raw_image.shape[0] / self.lbl_image.winfo_height()
                
                orig_x = int(x * scale_x)
                orig_y = int(y * scale_y)
                last_x = int(self.last_point[0] * scale_x)
                last_y = int(self.last_point[1] * scale_y)
                
                if self.draw_mode_var.get() == "free":
                    # Draw circle for free drawing
                    cv2.circle(self.mask_image, (orig_x, orig_y), 
                              self.brush_size, color=1, thickness=-1)
                else:
                    # Draw line for line mode
                    cv2.line(self.mask_image, 
                            (last_x, last_y),
                            (orig_x, orig_y), 
                            color=1, thickness=self.brush_size)
                
                self.last_point = (x, y)
                self.update_display()
            except Exception as e:
                print(f"Drawing error: {str(e)}")
    
    def stop_drawing(self, event):
        """Stop drawing mask"""
        self.drawing = False
        self.update_display()
    
    def apply_mask(self):
        """Create mask from edges and close window"""
        try:
            import cv2
            # Convert display image to numpy array
            edges = np.array(self.display_image)
            
            # Threshold to create binary mask
            _, self.mask_image = cv2.threshold(edges, 128, 1, cv2.THRESH_BINARY)
            
            # Save current mask with unique name
            mask_name = f"mask_{len(self.masks)+1}"
            self.masks[mask_name] = self.mask_image
            print(f"Saved mask: {mask_name}")
            
            # Show statistics
            self.show_statistics()
            
            # Close window
            self.root.destroy()
        except Exception as e:
            print(f"Mask creation failed: {str(e)}")
            
    def show_statistics(self):
        """Display statistics for selected region"""
        if self.mask_image is None or self.raw_image is None:
            return
            
        # Calculate statistics
        masked_data = self.raw_image[self.mask_image == 1]
        
        if len(masked_data) == 0:
            return
            
        # Create statistics window
        self.stats_window = tk.Toplevel(self.root)
        self.stats_window.title("Region Statistics")
        
        # Create figure for histogram
        fig = plt.Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        
        # Plot histogram
        ax.hist(masked_data, bins=256, range=(0,65535), 
               log=True, histtype='stepfilled')
        
        # Calculate statistics
        stats = {
            "Mean": np.mean(masked_data),
            "Median": np.median(masked_data),
            "Std Dev": np.std(masked_data),
            "Min": np.min(masked_data),
            "Max": np.max(masked_data),
            "25th Percentile": np.percentile(masked_data, 25),
            "75th Percentile": np.percentile(masked_data, 75),
            "Pixel Count": len(masked_data),
            "Area (px)": np.sum(self.mask_image),
            "Area (%)": np.sum(self.mask_image)/self.mask_image.size*100
        }
        
        # Add statistics text
        stats_text = "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in stats.items()])
        
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title("Selected Region Histogram (Log Scale)")
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Pixel Count (log)")
        
        # Display in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.stats_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        # Add close button
        close_btn = tk.Button(self.stats_window, text="Close", 
                            command=self.stats_window.destroy)
        close_btn.pack(pady=10)
            
    def show_statistics(self):
        """Display statistics for selected region"""
        if self.mask_image is None or self.raw_image is None:
            return
            
        # Calculate statistics
        masked_data = self.raw_image[self.mask_image == 1]
        
        if len(masked_data) == 0:
            return
            
        # Create statistics window
        self.stats_window = tk.Toplevel(self.root)
        self.stats_window.title("Region Statistics")
        
        # Create figure for histogram
        fig = plt.Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        
        # Plot histogram
        ax.hist(masked_data, bins=256, range=(0,65535), 
               log=True, histtype='stepfilled')
        
        # Calculate statistics
        stats = {
            "Mean": np.mean(masked_data),
            "Median": np.median(masked_data),
            "Std Dev": np.std(masked_data),
            "Min": np.min(masked_data),
            "Max": np.max(masked_data),
            "25th Percentile": np.percentile(masked_data, 25),
            "75th Percentile": np.percentile(masked_data, 75),
            "Pixel Count": len(masked_data),
            "Area (px)": np.sum(self.mask_image),
            "Area (%)": np.sum(self.mask_image)/self.mask_image.size*100
        }
        
        # Add statistics text
        stats_text = "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in stats.items()])
        
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title("Selected Region Histogram (Log Scale)")
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Pixel Count (log)")
        
        # Display in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.stats_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = Tiff16Viewer(root)
    root.mainloop()