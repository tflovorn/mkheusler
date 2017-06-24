# Dependencies and installation

Add to .bashrc:

    export CPATH=$HOME/local:$CPATH
    export LIBRARY_PATH=$HOME/local:$LIBRARY_PATH

Restart shell to set these variables.

Get libyaml and make:

    cd ~
    wget http://pyyaml.org/download/libyaml/yaml-0.1.7.tar.gz
    cd yaml-0.1.7
    ./configure --prefix=$HOME/local
    make
    make install

Install mkheusler:

    python3 setup.py develop --user
