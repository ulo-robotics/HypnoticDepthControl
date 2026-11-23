"""
Edge deployment: the controller and the classical baselines on embedded boards and accelerators, built and timed on the boards themselves.

  runner     the command line (edge.py): prepare (here), run (over ssh, or locally), table
  baselines  UKF-MM, MPC-MM's and MHE+NMPC's NLPs as CasADi-generated C, and their reference closed loops
  trt        the ONNX step for TensorRT 8.x (Jetson, JetPack 4: opset 13, batch 1)
  coral      the split policy for a Coral Edge TPU: int8 dense layers on the TPU, the rest in float on the host
The board-side sources are in edge/ (baselines, tensorrt, coral) and cpp/ (the C++ engine); edge/README.md lists what each board needs.
"""
