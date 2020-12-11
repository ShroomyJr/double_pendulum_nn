# Physics-Assisted Double Pendulum Neural Network
A multi-layer perceptron predicting the position of a double pendulum

## Abstract
A double pendulum is a simple physical system consisting of a pendulum with another pendulum attached to its un-fixed end. However, this simple system can result in chaotic movement that is difficult and computationally expensive to predict. I propose using a physics assistant neural network to predict the movement of a chaotic double pendulum. 

A MPL neural network will be trained on a dataset of observed positions of a double pendulum under chaotic motion along with a mathematical model that predicts the motion of the pendulum as well. Properties such as the length, mass, and positions of each rod will be perceived the network as inputs to generate the predicted position of both pendulums on the next time step.

Backpropagation networks have been frequently discussed within the course over the previous few weeks, and I believe applying the basic practices demonstrated through the course to the double pendulum will be very fitting. The double pendulum provides a simple model that can be visualized easily, but provides a complex and rich movement that should hopefully show that the neural network is able to strongly approximate the movement of a real life pendulum.

## Data Set
The network will be trained using a IBM's "Double Pendulum Chaotic" dataset that features position data measured from various real-life recordings of a double pendulum. You can see a gif from one of these recordings below:

![Real Double Pendulum Gif](./IMG/ibm_irl_pendulum.gif)

> The movement slowed down greatly when converting this gif.

While the data set is generated from a series of videos, the network will be trained using a collection of CSV files containing x and y position data for the pendulum. In addition to this real-life data set, I'll also be using a physics-based model to predict the pendulum. The physics based model will be derived and calculated beforehand for each test data set. This data set will be used to add an additional term to the loss function: the physics-based error.

The network will be trained to minimize the total error between the actual dataset and the error with physics based model.

## Plotting
To visualize the csv data, the `make_plot.py` file can be used to generate an animated plot showing the movement of the pendulum. This will be useful for comparing the movement of the actual double pendulum, the physics-based model, and the network's model. An example of the visualization can be seen below:

![Animated Double Pendulum Gif](./IMG/pendulum_0_rotated.gif)

> While plotting, it's become apparent that the data needed to be rotated 90 degrees to match the original footage

## Deriving the Physics-guided model

As the double pendulum is a simple physical system, very little information is required to derive the anticipated position of the pendulum.

The original dataset provided by IBM captures pixel-position data for the three points on the double pendulum. To calculate how the pendulum will continue moving from any given point we need six values:

1.  θ<sub>1</sub>: angle between limb 1 and the vertical axis
2.  θ<sub>2</sub>: angle between limb 2 and the vertical axis
3.  M<sub>1</sub>: mass of bob 1
4.  M<sub>2</sub>: mass of bob 2
5.  v<sub>1</sub>: speed of the center of mass of bob 1
6.  v<sub>2</sub>: speed of the center of mass of bob 2

To obtain the angles from our original position data, we first obtain the length of both pendulum arms through:

-   L<sub>1</sub> = dist{(x<sub>1</sub>, x<sub>2</sub>), (y<sub>1</sub>, y<sub>2</sub>)}
-   L<sub>2</sub> = dist{(x<sub>2</sub>, x<sub>3</sub>), (y<sub>2</sub>, y<sub>3</sub>)}

From here we can calculate θ<sub>1</sub> & θ<sub>2</sub>:

-   θ<sub>1</sub>: arcsin((y<sub>2</sub> - y<sub>1</sub>)/L<sub>1</sub>) + Π/2
-   θ<sub>2</sub>: arcsin((y<sub>2</sub> - y<sub>1</sub>)/L<sub>1</sub>)

To generate the data set that will be used for training the network, we'll use `scipy`'s integrate function to derive the momenta and theta values for each time stamp. After having generated the theta values for each time interval, they can be translated back into x and y values in the same form as the original real life dataset.

## The Neural Network

The design of the network will be a recurrent backpropagation network. For each time step, the outputs of the network will become the new inputs. Additionally, two nodes for mass will be used when training the network. Due to an inability to model real world double pendulums' mass in a way that is easily modellable, I hope that this will allow the network to train towards an accurate representation of the real world pendulum. These mass values will hopefully be able to be ignored using they're constant weight over the period as a sort of dynamic bias value. A diagram of the proposed network can be seen below:

![NN Map](./IMG/network_diagram.png)