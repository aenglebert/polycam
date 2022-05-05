# Polycam 
  

## 1. Introduction 

Implementation of the PolyCAM methods: PCAM+, PCAM- and PCAM+/- in pytorch.

Python scripts are provided to generate and evaluate the multiples saliency map computation methods (ours and previous methods). 

Jupyter notebooks are provided to get visual feedback. 


## 2. Requirements   

A requirements.txt file is provided for the required python libraries for our methods. 

The validation set of ILSVRC2012 is required for the scripts and most of the jupyter notebooks, the images must be stored in a folder, named "images" by default   (Can be changed with --image_folder option). 

The list of selected images should be provided in a txt file (the list of 2000 randomly selected images used for the paper is provided by default in "images.txt") 

The saliency maps can be either generated using the provided scripts or downloaded from the link https://polycam.ddns.net and are assumed by default to be in a folder named "npz". 

The faithfulness measurements can be generated using the scripts or downloaded from https://polycam.ddns.net. The files are assumed by default to be in a folder named "csv". 

Note that optional dependencies could be required for specific notebooks (mentioned at the beginning of the notebook if required), e.g., widget for a GUI interactive visualization 

For comparison with previous works, the method to compare with should be installed 

Torchcam: gradcam, gradcampp, smoothgradcampp, scorecam, sscam, iscam  
    if needed: pip install torchcam 

RISE,
    if needed: git clone https://github.com/eclique/RISE 

Captum: InputXGradient (ixg), IntegratedGradient (ig), SmoothGrad (sg), occlusion 
    if needed: pip install captum   

ZoomCAM:
    if needed git clone https://github.com/X-Shi/Zoom-CAM zoomcam


## 3. Library usage 

You can import the methods in your own script if desired.
PCAM+ (PCAMp), PCAM- (PCAMm) and PCAM+/- (PCAMpm) are availables.

example of usage: 
``` 
# Define your pytorch model here 
from torchvision.models import resnet50 

model = resnet50(pretrained=True) 
model.eval() 


# Initialize the saliency class
from polycam.polycam import PCAMpm
pcampm = PCAMpm(model)

# Set tranforms, normalise to ImageNet train mean and sd 
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((224, 224)), 
                                transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                                ]) 

# Get image 
from PIL import Image 
image = Image.open(path/to/image.png).convert('RGB') 
input_tensor = transform(image) 
  

# Generate saliency map (rem: [-1] is used to get the higher resolution saliency map, using the most layers, in a list of intermediate maps) 
saliency_map = pcampm(input_tensor)[-1] 
``` 


## 4. Notebooks 

Visual notebooks are provided for visualizations. They are made to be opened in jupyter. 

### 4.1 Interactive saliency map visualizations 

The file interactive_saliency_visualization.ipynb is providing a notebook with interactive GUI to visualize saliency maps generated and stored in the npz files. 

### 4.2 Metrics   

The file 'metrics.ipynb' provide an easy visualization of the faithfulness metrics, either as curves for the dataset or specific image, but also as mean auc of metrics. 

### 4.3 Sanity check 

A notebook is provided to generate the sanity check for the PolyCAM methods 


## 5. Scripts 

Ready to use scripts are provided for the methods we compare to in the paper 

### 5.1 Saliency map generations 

The file generate_explanations.py can be used to generate explanations for the images list. Output the saliency maps in a npz file.
Saliency generations methods are named in minuscules for the scripts.

example of usage: 
``` 
python generate_explanations.py --model vgg16 --saliency pcampm --cuda --batch_size 4 
``` 

for more details on parameters: 
``` 
python generate_explanations.py --help  
``` 

By default, PCAM+ (pcamp), PCAM- (pcamm) and CPAM+/- (pcampm) are readily available. 

Optionally, the script can generate saliency maps for other methods with the corresponding libraries (see Requirements above) 
  

### 5.2 Faithfulness metrics 
  
The file evaluation.py can be used to generate the Faithfulness metrics for any saliency maps using the npz file generated by generate_explanations.py 
The insertion and deletion metrics are computed for each image and stored in csv files as detailed values for each step and as auc. 
They can be easily visualized using the dedicated jupyter notebook. 

example of usage: 
``` 
python evaluation.py --model vgg16 --saliency_npz npz/vgg16_pcampm4.npz --cuda 
``` 

for more details on parameters: 
``` 
python evaluation.py --help  
```

nb: please check you use the same model in the generation of the saliency maps and the evaluation


### 5.3 Sensistivity analysis 

The file sensitivity.py can be used to generate the sensitivity max metric for the images list. The metrics are saved in a csv file that can be read with metrics.ipyn.

example of usage: 
``` 
python sensitivity.py --model vgg16 --saliency pcampm --cuda --batch_size 4
``` 

for more details on parameters: 
``` 
python sensitivity.py --help  
``` 

By default, PCAM+ (pcamp), PCAM- (pcamm) and CPAM+/- (pcampm) are readily available. 

Optionally, the script can generate saliency maps for other methods with the corresponding libraries (see Requirements above) 
  

