# LLM Exploration Summary

## Session Focus

The primary focus of this session was optimizing the learning rate and convergence, explaining why the loss was plateauing, and transitioning the training process from CPU to GPU to accelerate development.

## Surprising Insights

Conversation: Troubleshooting stagnant loss during reconstruction.
Things that surprised me: When moving the process to CUDA, the loss initially stayed exactly the same across epoch. This indicated that even though the code was running on the GPU, the optimizer wasn't actually updating the parameters because the variable c hadn't been re-initialized as a leaf tensor on the new device with requires_grad=True.

## Techniques that worked

Re-initializing the Optimizer on the Device: Moving the parameter c to the GPU before passing it to the Adam optimizer was crucial. If you move a tensor after the optimizer is created, the optimizer still holds a reference to the old CPU memory address.
Dynamic Learning Rate Adjustments: Using a Gamma=0.99 scheduler helped initially, but manually observing the loss curves showed that a higher starting learning rate (0.01) with Adam was more effective than a decaying rate that was too conservative.
Explicit Floating Point Casts: Using .float() during the device transfer ensured that Phi and the target tensors were in Float32, preventing precision errors that can cause the loss to stall.

## Dead Ends Worth Noting

Approaches that seemed promising but failed 1: Using ExponentialLR with a high decay rate too early.
How I realized it wouldn’t work: The loss "bottomed out" at 700 and stopped moving because the learning rate became so small that the gradients couldn't push the weights out of a local minimum.
Approaches that seemed promising but failed 2: Simple device transfer like c.to(device).
How I realized it wouldn’t work: In PyTorch, tensor.to(device) on a tensor with gradients can break the computational graph if not handled carefully. I had to explicitly initialize c on the device to ensure it remained a "leaf" tensor for the autograd engine.

## Next Steps

The model is now converging much better, breaking past the previous barrier of 700 and heading toward 400. The next step is to see if the numerical drop in loss actually corresponds to a sharper image, or if we are just over-fitting the noise in the k-space data. I will also experiment with Learning Rate Warmup to see if we can reach convergence even faster.
