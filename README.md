# There's Waldo!

Built a model that is able to find Waldo in a Where's Waldo map with relative success.

The model was designed using Tensorflow and trained using images from various sources. 

The trained model performs a convolution with a set stride on the map and outputs a probability map which is then fed through a probability function to determine how likely it is that it is looking at Waldo. Then the analysed map is overlaid onto the original map and displayed.

The entire process takes about 7-8 seconds to run, which is considerably less time than an average human would take to find Waldo.

Below are examples of maps where the model was able to locate Waldo:

![](https://github.com/antoniok9130/WheresWaldo/blob/master/Examples/Map%201.png)
![](https://github.com/antoniok9130/WheresWaldo/blob/master/Examples/Map%203.png)
![](https://github.com/antoniok9130/WheresWaldo/blob/master/Examples/Map%206.png)
![](https://github.com/antoniok9130/WheresWaldo/blob/master/Examples/Map%207.png)
![](https://github.com/antoniok9130/WheresWaldo/blob/master/Examples/Map%209.png)
![](https://github.com/antoniok9130/WheresWaldo/blob/master/Examples/Map%2014.png)
![](https://github.com/antoniok9130/WheresWaldo/blob/master/Examples/Map%2019.png)
