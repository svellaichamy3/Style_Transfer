## Introduction
Wouldn't it be amazing to get Picasso or Van Gogh to paint your beautiful neighbourhood in their own style? Deep Learning helps us do that! We take an image and add the style of another reference style image to it and give it a new look. We do this experiment inspired from ”Image Style Transfer Using Convolutional Neural Networks” (Gatys et al CVPR 2015). 

## Idea
The general idea is to take two images (a content image and a style image), and produce a new image that reflects the content of one but the artistic ”style” of the other. We will do this by first formulating a loss function that matches the content and style of each respective image in the feature space of a deep network, and then performing gradient descent on the pixels of the image itself. In this project, we use SqueezeNet as our feature extractor.

We can generate an image that reflects the content of one image and the style of another by incorporating both in our loss function. We want to penalize deviations from the content of the content image and deviations from the style of the style image. We can then use this hybrid loss function to perform gradient descent not on the parameters of the model, but instead on the pixel values of our original image.

<p align="center">
<img src="https://github.com/svellaichamy3/Style_Transfer/blob/main/images/style3_wm.png" />
</p>



### Content Loss
Content loss measures how much the feature map of the generated image differs from the feature map of the source image. We only care about the content representation of one layer of the network (say, layer _l_) that has feature maps A<sup>_l_</sup> &#1013; R<sup>1*C<sub>_l_</sub>*H<sub>_l_</sub>*W<sub>_l_</sub></sup>. _C<sub>l</sub>_ is the number of channels in layer _l_, _H<sub>l</sub>_ and _W<sub>_l_</sub>_ are the height and width. We will work with reshaped versions of these feature maps that combine all spatial positions into one dimension. Let F<sup>_l_</sup> &#1013; R<sup>N<sub>_l_</sub>*M<sub>_l_</sub></sup> be the feature map for the current image and P<sup>_l_</sup> &#1013; R<sup>N<sub>_l_</sub>*M<sub>_l_</sub></sup> be the feature map for the content source image where M<sub>_l_</sub> = H<sub>_l_</sub>* W<sub>_l_</sub> is the number of elements in each feature map. Each row of F<sup>_l_</sup> or  P<sup>_l_</sup> represents the vectorized activations of a particular filter, convolved over all positions of the image. Finally, let w<sub>c</sub> be the weight of the content loss term in the loss function. Then the content loss is given by:</n>
<p align="center">
<img src="https://github.com/svellaichamy3/Style_Transfer/blob/main/images/contentlosseqn.PNG" />
</p>

### Style Loss
Now we can tackle the style loss. For a given layer _l_, the style loss is defined as follows:
First, compute the Gram matrix G which represents the correlations between the responses of each filter, where F is as above. The Gram matrix is an approximation to the covariance matrix – we want the activation statistics of our generated image to match the activation statistics of our style image, and matching the (approximate) covariance is one way to do that. There area variety of ways you could do this, but the Gram matrix is nice because it’s easy to compute and in practice shows good results. Given a feature map F<sup>_l_</sup> of shape (1,C<sub>_l_</sub>,M<sub>_l_</sub>), the Gram matrix has shape (1,C<sub>_l_</sub>,C<sub>_l_</sub>) and its elements are given by:

<p align="center">
<img src="https://github.com/svellaichamy3/Style_Transfer/blob/main/images/stylelosseqn1.PNG" />
</p>

Assuming G<sup>_l_</sup> is the Gram matrix from the feature map of the current image, A<sup>_l_</sup> is the Gram Matrix from the feature map of the source style image, and w<sub>_l_</sub> a scalar weight term, then the style loss for the layer _l_ is simply the weighted Euclidean distance between the two Gram matrices:

<p align="center">
<img src="https://github.com/svellaichamy3/Style_Transfer/blob/main/images/stylelosseqn2.PNG" />
</p>

In practice we usually compute the style loss at a set of layers L rather than just a single layer _l_; then the total style loss is the sum of style losses
at each layer:

<p align="center">
<img src="https://github.com/svellaichamy3/Style_Transfer/blob/main/images/stylelosseqn3.PNG" />
</p>

### Total Variation Loss
It turns out that it’s helpful to also encourage smoothness in the image. We can do this by adding another term to our loss that penalizes wiggles or **total variation** in the pixel values. This concept is widely used in many computer vision task as a regularization term. You can compute the ”total variation” as the sum of the squares of differences
in the pixel values for all pairs of pixels that are next to each other (horizontally or vertically). Here we sum the total-variation regualarization for each of the 3 input channels (RGB), and weight the total summed loss by the total variation weight, wt:

<p align="center">
<img src="https://github.com/svellaichamy3/Style_Transfer/blob/main/images/tvlosseqn.PNG" />
</p>

## Results
 It was a wonderful fun experiment where we captured the style of one image and transferred it to the content of another image. Here are a few more results:
 <p align="center">
<img src="https://github.com/svellaichamy3/Style_Transfer/blob/main/images/style2_wm.png" />
</p>

<p align="center">
<img src="https://github.com/svellaichamy3/Style_Transfer/blob/main/images/style1_wm.png" />
</p>
