Dataset: CIFAR10 Model: VGG19

Category: Image classification

To start with..


Unzip File CS260finalRB.zip 
The zip file unzips into two directories 

 VGGPruning - 
              traincifar10.py # Train the original VGG19 network 
              vgg.py # module definition 
              util.py # monitor run time progress 
              prune.py # Train and test VGG16 with pruning 
              prune.sh # script to run prune.py with different input thresholds (0.75, 0.5, 0.25, 0.1) 
              VGGsize.txt - VGG19 original model size 
              trainlog.txt - log file for original traincifar10.py run 
              prune_75log.txt, prune_05log.txt, prune_25log.txt # log file with pruning thresholds 
