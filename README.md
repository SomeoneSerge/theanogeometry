![Scheme](logo/stocso31.jpg)

# Theano Geometry #

### Who do I talk to? ###

Please contact Stefan Sommer *sommer@di.ku.dk*

### Installation Instructions ###

#### Linux:
Install numpy, scipy, theano, jupyter, matplotlib, multiprocess, sklearn:
```
pip install numpy scipy theano jupyter matplotlib multiprocess sklearn
```
Use e.g. a Python 3 virtualenv:
```
virtualenv -p python3 .
source bin/activate
pip install numpy scipy theano jupyter matplotlib multiprocess sklearn
```
Set theano options in ~/.theanorc, e.g.
```
    [global]
    floatX = float64
    device = cpu
    optimizer = fast_run
    cast_policy = numpy+floatX
```
Start jupyter notebook as in
```
    export OMP_NUM_THREADS=1; jupyter notebook
```

Some features, e.g. higher-order landmarks, may require a 'Bleeding-Edge Installation' installation of Theano, see http://deeplearning.net/software/theano/install.html installation instructions.


#### Linux - vagrant:
Install vagrant and virtualbox, on Ubuntu e.g.:
```
apt install vagrant virtualbox
```
In the vagrant folder, run vagrant and ssh to the box:
```
vagrant up
vagrant ssh -- -L 8888:localhost:8888
```
Open the url http://localhost:8888/ in your web browser. The notebook password is '12345'

#### Windows:
Theano Geometry can be conveniently used on Windows with the provided Vagrantfile:

Step-by-step guide for Windows users

Premise:

We are going to use a Virtual Machine, so we need to check that our actual machine allows for it to work. You might want to enter your BIOS and check if virtualization is enabled. Otherwise, just ignore this premise. You can always go back to it if the need arises.
An example procedure for enabling virtualization for Lenovo laptops is detailed at: https://support.lenovo.com/dk/en/solutions/ht500006

1. Download and install Vagrant from https://www.vagrantup.com/downloads.html		
1. Download and install Virtualbox from https://www.virtualbox.org/wiki/Downloads
1. Download and install a SSH client from http://sshwindows.sourceforge.net/download/		
1. Save the file 'Vagrantfile' in the vagrant directory
1. Start a command prompt (e.g. Search for 'cmd')						
1. Change directory ('cd') to the directory where you saved the Vagrantfile						
1. Run 'vagrant up' and wait a while until the command has finished. If you get an error message stating that "VT-x is disabled", then you need to enter your BIOS and enable virtualization (see premise above).
1. Run 'vagrant ssh -- -L 8888:localhost:8888'. 
1. When asked for "passphrase for key", press Enter. When asked for password, enter: 12345
1. Fire up your favourite web browser and open 'http://localhost:8888/'
1. Enter password '12345' and log in.