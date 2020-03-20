
Overview
The project explores the compression techniques for neural nets. Specifically, it takes VGG19 and apply a basic pruning method by trimming weights across the layers in CNN architecture. Additionally it also explores the squeeze net algorithm as a second method by squeezing the layers, improving the activations maps and slimming the front end filters and input channels so to reduce the size of the network.  

How to Start?
•	Unzip the file CS260finalRB.zip – It unzips into following two folders
VGGPruning - 
               traincifar10.py # Train the original VGG19 network 
               vgg.py # module definition 
         	     util.py # monitor run time progress 
      	        prune.py # Train and test VGG16 with pruning 
               prune.sh # script to run prune.py with different input thresholds (0.75, 0.5, 0.25, 0.1) 
               VGGsize.txt - VGG19 original model size 
               trainlog.txt - log file for original traincifar10.py run 
               prune_75log.txt, prune_05log.txt, prune_25log.txt # log file with pruning thresholds
VGGSqueezenet - 
               train_s.py # Train the original VGG19 network 
               vgg_s.py # original module definition 
         	     util_s.py # monitor run time progress 
      	        vgg_sqz.py # squeeze net with fire layers 
		             train_with_sqz.py # training with squeeze model vgg_sqz
