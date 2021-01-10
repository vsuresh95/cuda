# cuda-crashcourse

This repository contains the code samples executed while learning CUDA

## Compile steps
```
cd <test name>
```
Deleting all generated files:
```
make clean
```
Compile and execute the test without options:
```
make
```
Compile and execute the test with options:
```
make OPTS='<enter all argv[] options>'
```
Compile, generate debug files and execute the test with options:
```
make all_debug OPTS='<enter all argv[] options>'
```
Create a new test folder (run this from root of repository):
```
make create_test FOLDER_NAME=<enter test name>
```
