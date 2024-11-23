# Swedish Complexity Text Generation #
>LT2326 final project implementation

To run at the current device:  ```python CanRun_currentCUDA_swedish_text_generator.py```

To run at the designated device (e.g. cuda:2):  ```python CanRun_manualCUDA_swedish_text_generator.py```

To revise device number, change the last line here (line 236 in the "CanRun_manualCUDA_swedish_text_generator.py") to the device number:
>import os

>os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

>os.environ["CUDA_VISIBLE_DEVICES"]="2" # for cuda:2
