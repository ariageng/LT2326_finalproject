# Swedish Complexity Text Generation #
>LT2326 final project implementation

To run at the current device:  ```python CanRun_currentCUDA_swedish_text_generator.py```
To run at the designated device (e.g. cuda:2):  ```python CanRun_manualCUDA_swedish_text_generator.py```
>import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
