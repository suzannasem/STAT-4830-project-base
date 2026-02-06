# Week 4 Project Report: MRI Reconstruction

## Problem Statement
MRI quality is constrained by scan time. Higher-resolution and lower signal to noise ratio allow for a clearer view of tissues, but require longer scan time. This can be uncomfortable for patients and increase hospital operating costs. Improving image reconstruction from undersampled data enables shorter scan times while preserving image quality benefits both patients and hospitals.

MRI scanners don’t measure images in pixel space, but instead measure values in k-space. K-space contains Fourier coefficients of the image. Fully sampled k-space allows for exact reconstruction, but in practice k-space is undersampled to shorten scans. This missing information makes image reconstruction a non-injective inverse problem, requiring optimization techniques to recover the most accurate image.

In this project, we reconstruct the image in a kernel-defined function space. The image is represented as a linear combination of kernel basis functions evaluated on the image grid. The optimization variables are the kernel coefficients $\alpha\in\mathbb{R}^n$. We optimize these coefficients by minimizing the difference between the predicted and measured k-space values. 

Reconstruction quality will be evaluated both visually and using quantitative metrics, such as mean squared error and peak signal-to-noise ratio. Key constraints include restricting reconstructions to the span of the kernel basis and recovering under k-space undersampling. Potential failure modes include convergence to local minima and overfitting to noise. 

## Technical Approach
We mathematically represent the MRI model by the linear operator $A=MF$, where $F$ denotes the discrete Fourier transform mapping the image to k-space and $M$ is a diagonal matrix with entries 0 or 1; a value of 1 means the frequency is being measured and a value of 0 means it is not. Given observed undersampled k-space data $y_{obs}$, we solve for $\alpha$ such that $\min_{\alpha\in R^n}||A(K\alpha)-y_{obs}||_2^2$.

Direct pixel-space reconstruction for undersampled MRI is ill-conditioned because  is not injective, meaning many distinct images can fit the same measured data. To address this, we restrict the reconstruction to a structured hypothesis (or class) defined by kernel basis functions. This removes directions that cannot be reliably inferred from the measurements. We use Gaussian kernels because they impose smoothness and locality, which are natural structural assumptions for MRI images and help control high-frequencies. 

We used data from the UPenn-GBM Cancer Imaging Archive, which is open source and can be accessed with the NBIA data retrieval system. This dataset includes full stacks of 2D MRI images: we selected an image at the midpoint as our target. To create our kernel basis, we first created a 2D grid in PyTorch which we designated as the space of possible kernel bases, then calculated Euclidean distance between each pixel and the origin, and solved for weights of Gaussian kernels, converting the data into more easily readable forms. We convert the kernel basis from 2D into a frequency in k-space form. Since the data we are using consists of pre-constructed MRI images, we simulate the under-sampling of frequencies that is typical in MRI by ‘masking’ the data, adding a filter to only show low-frequency waves. To solve the optimization problem, we use an Adam (Adaptive Moment Estimation) optimizer, as it efficiently handles noise. The two primary methods of validation are (1) comparison to original MRI target image and (2) measurement of loss over iterations.

## Initial Results
Here is an image of our first reconstruction using what we have so far:
{}

We achieved a PSNR of 22.78 dB and an MSE of 0.00529, although it is unclear how “good” these benchmarks are and a baseline is necessary for comparison to see if our method makes improvements. 

Current limitations include that synthetic under-sampling of k-space may not accurately reflect actual undersampling that occurs during quick MRI scans. The accuracy of the reconstruction would change depending on the number of pixels included in the sample. 

Our resource usage measurements can be seen here:
{}

We were successful in loading 120 MRI image slices and converting the middle slice to a 256x256 pixel grid. Since we used a single image for our optimization, we were able to simulate ten thousand epochs without large runtime effects. In practice, physicians would typically be combining multiple low-resolution MRI images into a single high-resolution one using these reconstruction methods: using multiple images or even an entire stack would introduce computational constraints as the filesize increases.

## Next Steps
In terms of next steps, for immediate improvements, we have to focus on decreasing loss to bring the reconstructed image closer to the target. Additionally, we have to tackle filesize constraints as the multiple slices take up a lot of memory. While we used a Gaussian kernel to ensure smoothness across the image, there are other possible kernels we can try. The LaPlace kernel, for example, provides sharper boundaries which may be more helpful in a MRI context, where irregularities can be as small as a few pixels. Imposing a TV (total variation) penalty would also help preserve sharp edges while regularizing.

We may have to consider broadening the scope of the project as well since it is difficult for us to make an accurate benchmark of success, usability, and accuracy of the reconstructed MRI image without the adequate medical background. This may look like broadening to more general image reconstruction and using data where we have a stronger understanding of what is ‘correct’ and better benchmark the progress of our algorithm. To add a baseline for validation, we can implement zero-filled FFT, which fills all missing values with a value of 0. PSNR relative to ground truth is what is implemented currently, but it is hard to interpret whether the reconstruction error is small or large. Comparing this to zero-filled FFT provides a baseline for the PSNR scale to see how much our implementation actually provides as opposed to the trivial reconstruction. 


