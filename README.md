# DiffusionVideoEditing
Official project repo for paper "Speech Driven Video Editing via an Audio-Conditioned Diffusion Model" 

Shoutout to https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models ! Most of the code in this repo is taken from there. It's a really good implementation of the Palette image2image paper, so go check it out! 

You can check out some of the results on the project page found here: https://danbigioi.github.io/DiffusionVideoEditing/

And if you want to read the paper, this is the link to the arxiv submission: https://arxiv.org/abs/2301.04474 

Note that while the work presented in this paper is relatively early stage stuff, i'll be using this repo as a "base" of sorts where I'll be periodically updating the model and implementation with newer approaches im currently working on. 

# UPDATE: 

Hey all, sorry for the radio silence on this! Got some reviews on the original paper and have been busy addressing them and improving the original implementation. Happy to say that we are getting very nice results on single speaker now and are currently in the middle of evaluating for multi-speaker. Check out the video attached below :) Once I finish writing the udpated paper with all the fancy new improvements, ill add in the latest code too. If you're curious for how it works though, feel free to shoot me an email, always am happy to discuss.  

https://user-images.githubusercontent.com/44242832/222735254-f1880950-0d92-49e3-b447-ee3c8d3008e7.mp4

https://user-images.githubusercontent.com/44242832/222736036-9fa83652-9d7b-4693-a1fe-62719e3f78fd.mp4



# TO DO: 

Write some tutorials for how to use the code (I apologise its genuinely a mess, I need to streamline it a bit when I get the time). It's on my todo list to write out a detailed guide for both training and running inference. Also need to upload the trained model weights. 

Upload datasets. I'll see if I can somehow upload the whole processed dataset here, but most likely I'll upload the preprocessing scripts I made with a guide on          how to extract the stuff you need. For reference, I am using the GRID dataset for a lot of these experiments, but in theory the model should work with any                audiovisual dataset like TCD Timit, Lombard Grid, etc.  


