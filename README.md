# source-separation
Implementation of novel speech enhancement network architecture and time domain source separation network for multi-speaker signals.

The speech enhancement network leverages a complex U-Net design and demonstrates the ability to perform simultaneous echo cancellation and background noise removal. A convolutional TasNet is trained to receive the enhanced signals to separate the speaker mixtures into different channels. The U-Net design is a complex extension of the real valued network proposed by Kothapally et al., 2020. The ConvTasNet mimics the network design proposed by Luo et al., 2018.

The repository is currently limited to signals with two speakers or less.

The ,ipynb contains a walkthrough of the different sections of codes and the set up for experiments run on a number of configurations.

This project has gone through some refactoring so some bugs may be present. Please reach out to me if you have any questions or would like access to model weights. 
