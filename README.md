# Learning-Based Sensor Calibration for Photoacoustic Tomography

## Overview

This project investigates a learning-based approach to calibrating sensor-position errors in Photoacoustic Tomography (PAT). A lightweight neural network was developed to estimate and correct the effects of displaced sensors, with the goal of improving photoacoustic image reconstruction quality.

The project was conducted as part of a research collaboration at Rice University.

## Problem

Photoacoustic tomography reconstructs images from measurements collected by an array of sensors. The reconstruction quality depends on accurate knowledge of sensor positions.

In practical settings, sensors may be displaced from their intended locations due to positioning or manufacturing variations. This project explores whether a neural network can learn to calibrate these sensor-position errors and improve the resulting image reconstruction.

## Approach

The project follows a learning-based sensor calibration pipeline:

1. Generate photoacoustic data under ground truth sensor-position perturbations using standard tomographic mathematics.
2. Process the perturbed measurements and prepare them as neural network inputs.
3. Train a lightweight neural network to estimate the required calibration.
4. Apply the learned calibration to the sensor data.
5. Reconstruct the photoacoustic image and evaluate reconstruction quality.

### Technologies

- Python
- JAX
- Neural Networks
- Tomography
- Photoacoustic Tomography (PAT)
- Image Reconstruction

## Implementation

## Results

The learned calibration improved photoacoustic image reconstruction quality compared with the uncalibrated baseline.

| Method | SSIM |
|---|---:|
| Uncalibrated | 0.820 |
| Calibrated | 0.864 |

This corresponds to an improvement in SSIM from 0.820 to 0.864.

## Repository Structure

## My Contribution

## Poster

## Acknowledgments
