# BackgroundRemoverBot

2 ways to remove background using pure deep learning and its mixture with traditional methods

[Link to the telegram bot](https://t.me/simple_bg_remover_bot)

To work correctly, you need to add ```git clone https://github.com/shreyas-bk/U-2-Net``` to the project folder and *.pth* and *.h5* files with model weights in the models folder.

## Example of results:

### Original -> Pytorch model -> Tensorflow model

<p float="left">
  <img src="./examples/orig1.jpg" width="21%" />
  <img src="./examples/pytorch1.jpg" width="21%" />
  <img src="./examples/tensorflow1.jpg" width="21%" /> 
</p>

<p float="left">
  <img src="./examples/orig2.jpg" width="31%" />
  <img src="./examples/pytorch2.jpg" width="31%" />
  <img src="./examples/tensorflow2.jpg" width="31%" /> 
</p>

<p float="left">
  <img src="./examples/orig3.jpg" width="21%" />
  <img src="./examples/pytorch3.jpg" width="21%" />
  <img src="./examples/tensorflow3.jpg" width="21%" /> 
</p>

### Example of a picture that models can't handle:

<p float="left">
  <img src="./examples/citty.jpg" width="31%" />
  <img src="./examples/citty_pytorch.jpg" width="31%" />
  <img src="./examples/citty_tensorflow.jpg" width="31%" /> 
</p>
