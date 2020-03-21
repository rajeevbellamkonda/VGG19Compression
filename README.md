
# VGG Compression

# Overview
The project explores the compression techniques for neural nets. Specifically, it takes VGG19 and apply a basic pruning method by trimming weights across the layers in CNN architecture. Additionally, it explores the squeeze net algorithm as a second method by squeezing the layers, improving the activations maps and slimming the frontend filters and input channels so to reduce the size of the network. The inspiration and guidance for squeeze net has been derived from the paper published at this Link  

# Environment

·       Python 3

·       Pytorch

 

# How to Start?

·    * Unzip the file CS260finalRB.zip – It unzips into following two folders

     * source Code has been uploaded to [github](https://github.com/rajeevbellamkonda/VGG19Compression)

 

# Folder1:VGGPruning -

               traincifar10.py # Train the original VGG19 network

               vgg.py # module definition

               util.py # monitor run time progress

               prune.py # Train and test VGG16 with pruning

               prune.sh # script to run prune.py with different input thresholds (0.75, 0.5, 0.25, 0.1)

               VGGsize.txt - VGG19 original model size

               trainlog.txt - log file for original traincifar10.py run

               prune_75log.txt, prune_05log.txt, prune_25log.txt # log file with pruning thresholds

# Folder2:VGGSqueezenet -

               train_s.py # Train the original VGG19 network
               
               presqueezewithriginaVGGlog.txt # Log file with original VGG19
               
               vgg_s.py # original module definition

               util_s.py # monitor run time progress

               vgg_sqz.py # squeeze net with fire layers

               train_with_sqz.py # training with squeeze model vgg_sqz
               
               model_after_squeeze_log.txt # Log with squeeze net

# How to Run Pruning?
           

               Change Directory to VGGPruning

               python train_cifar10.py --net vgg # Train the original VGG19 model 
                           Arguments= lr = default 0.1
                                      net = default vgg
                                      r = default resume

               python prune.py --net vgg --prune 0.75 # TO Prune the original model with 75% weight reduction
                           Arguments= prune = default 0.5
                                      net = default vgg
                                      lr = 0.01
                                      

              ./prune.sh -- Shell script to run prune with various thresholds
 

# How to Run SqueezeNet?
             Change Directory to VGGSqueezenet

             python train_s.py # Train the original VGG19 model

             python train_with_sqz.py # Apply squeeze algorithm on original model
      
# Resources

  * [Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
  * [SqueezeNet](https://arxiv.org/abs/1602.07360)
  * [Quantization](https://pytorch.org/docs/stable/quantization.html)
  * [Github](https://github.com/kentaroy47/Deep-Compression.Pytorch)
             
